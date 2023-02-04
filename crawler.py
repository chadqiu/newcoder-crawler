# -*- coding: utf-8 -*-
import requests
import json
import time
import re
import pymysql
import smtplib
from email.mime.text import MIMEText


def _parse_newcoder_page(data, skip_words, start_date):
    assert data['success'] == True
    pattern = re.compile("|".join(skip_words))
    res = []
    for x in data['data']['records']:
        x = x['data']
        dic = {"user": x['userBrief']['nickname']}

        x = x['contentData'] if 'contentData' in x else x['momentData']
        dic['title'] = x['title']
        dic['content'] = x['content']
        dic['id'] = int(x['id'])
        dic['url'] = 'https://www.nowcoder.com/discuss/' + str(x['id'])

        if len(skip_words) > 0 and pattern.search(x['title'] + x['content']) != None:  # 关键词正则过滤
            continue

        createdTime = x['createdAt'] if 'createdAt' in x else x['createTime']
        dic['createTime'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(createdTime // 1000))
        dic['editTime'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x['editTime'] // 1000))

        if dic['editTime'] < start_date:  # 根据时间过滤
            continue
        res.append(dic)

    return res


def get_newcoder_page(page=1, keyword="校招", skip_words=[], start_date='2023'):
    header = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "content-type": "application/json"
    }
    data = {
        "type": "all",
        "query": keyword,
        "page": page,
        "tag": [],
        "order": "create"
    }
    x = requests.post('https://gw-c.nowcoder.com/api/sparta/pc/search', data=json.dumps(data), headers=header, )
    data = _parse_newcoder_page(x.json(), skip_words, start_date)
    return data


def upsert_to_db(data, host, user, passwd, database, charset, port):
    db = pymysql.connect(
        host=host,
        user=user,
        passwd=passwd,
        database=database,
        charset=charset,
        port=port
    )
    try:
        cursor = db.cursor()
        sql = "select id, edited_time from newcoder_search where id in ({})".format(
            ",".join([str(x['id']) for x in data]))
        cursor.execute(sql)
        exists = cursor.fetchall()
        dic = {x[0]: x[1].strftime("%Y-%m-%d %H:%M:%S") for x in exists}

        insert_data = [[x[k] for k in x] for x in data if x['id'] not in dic]
        update_data = [(x['editTime'], x['id']) for x in data if x['id'] in dic and dic[x['id']] != x['editTime']]
        sql = "INSERT INTO newcoder_search (user, title, content, id, url, created_time, edited_time) VALUES(%s, %s, %s, %s, %s, %s, %s)"
        cursor.executemany(sql, insert_data)
        sql = "update newcoder_search set edited_time = %s where id = %s"
        cursor.executemany(sql, update_data)
        db.commit()
    except Exception as e:
        print("db error: ", e)
    db.close()
    return [x for x in data if x['id'] not in dic], [x for x in data if
                                                     x['id'] in dic and dic[x['id']] != x['editTime']]


def _table_html_generate(data):
    s = '<table>'
    s += '<tr>' + "\n".join(["<th>" + x + '</th>' for x in data[0]]) + '</tr>'
    for d in data:
        s += '<tr>' + "\n".join(["<td>" + str(d[x]) + '</td>' for x in d]) + '</tr>'
    s += '</table>'
    return s


def send_email(insert_data, update_data, mail_host, mail_user, mail_pass, sender, receivers):
    msg = ''
    if len(insert_data) > 0:
        msg += '<h1>insert</h1></br>' + _table_html_generate(insert_data) + '</br></br>'
    if len(update_data) > 0:
        msg += '<h1>update</h1></br>' + _table_html_generate(update_data) + '</br></br>'
    if msg == '':
        msg = '<h1>今日无新增数据</h1></br>'

    message = MIMEText(msg, 'html', 'utf-8')
    message['Subject'] = '牛客网{}招聘信息'.format(time.strftime("%Y-%m-%d"))
    message['From'] = sender
    message['To'] = receivers[0]
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)
        # smtpObj.connect(mail_host, 465)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(
            sender, receivers, message.as_string())
        smtpObj.quit()
        return True
    except smtplib.SMTPException as e:
        print('email send error: ', e)
        return False


def run(keywords, skip_words, db_config, mail_config=None):
    res = []
    for key in keywords:
        print(key, time.strftime("%Y-%m-%d %H:%M:%S"))
        for i in range(1, 11):
            print(i)
            page = get_newcoder_page(i, key, skip_words)
            if not page:
                break
            res.extend(page)
            time.sleep(1)

    result, ids = [], set()  # 去重
    for x in res:
        if x['id'] in ids: continue
        ids.add(x['id'])
        result.append(x)

    print("total num: ", len(result))
    x = upsert_to_db(result, **db_config)  # insert_data, update_data
    if mail_config:
        send_email(*x, **mail_config)


def main():
    # 指定要过滤的词
    skip_words = ['求捞', '泡池子', '池子了', '池子中', 'offer对比', '总结一下', '给个建议', '开奖群', '没消息', '有消息', '拉垮', '求一个', '求助', '池子的',
                  '决赛圈', 'offer比较', '求捞', '补录面经', '捞捞', '收了我吧', 'offer选择', '有offer了', '想问一下', 'kpi吗', 'kpi面吗', 'kpi面吧']

    # 指定搜索的关键词
    keywords = ['补招', '补录']

    # 配置数据库信息
    db_config = {
        "host": "localhost",
        "user": "root",
        "passwd": "your password",
        "database": 'your database',
        "charset": 'utf8',
        "port": your mysql port
    }

    # 配置邮箱信息
    mail_config = {
        "mail_host": 'smtp server host',
        "mail_user": 'your user name',
        "mail_pass": 'password',  # 密码(部分邮箱为授权码)
        "sender": 'sender email',
        "receivers": ["receivers email"]
    }

    run(keywords, skip_words, db_config, mail_config)


if __name__ == "__main__":
    main()
    print("end")