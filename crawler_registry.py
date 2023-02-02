# -*- coding: utf-8 -*-
import requests
import json
import time
import re
import pymysql
import smtplib
import nacos
from email.mime.text import MIMEText

def get_newcoder_page(page = 1, keyword = "补录"):
    header = {
        "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "content-type":"application/json"
    }
    data = {
        "type": "all", 
        "query": keyword, 
        "page": page, 
        "tag": [], 
        "order": "create"
    }
    x = requests.post('https://gw-c.nowcoder.com/api/sparta/pc/search', data = json.dumps(data), headers = header, )
    return x.json()

def parse_newcoder_page(data, skip_words = [], start_date = '2023'):
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
        
        if skip_words and pattern.search(x['content'] + x['title'] if x['title'] else "") != None:
            continue
  
        createdTime = x['createdAt'] if 'createdAt' in x else x['createTime']
        dic['createTime'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(createdTime // 1000))
        dic['editTime'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x['editTime'] // 1000))
        
        if dic['editTime'] < start_date: continue
        res.append(dic)
        
    return res

def upsert_to_db(data, host, user, passwd, database,charset, port):
    db = pymysql.connect(
          host=host,
          user=user,
          passwd=passwd,
          database = database,
          charset=charset,
          port = port
    )
    try:
        cursor = db.cursor()  
        sql = "select id, edited_time from newcoder_search where id in ({})".format(",".join([str(x['id']) for x in data]))
        cursor.execute(sql)
        an = cursor.fetchall()
        dic = {x[0] : x[1].strftime("%Y-%m-%d %H:%M:%S") for x in an}

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
    return [x for x in data if x['id'] not in dic], [x for x in data if x['id'] in dic and dic[x['id']] != x['editTime']]

def table_html_generate(data):
    s = '<table>'
    s += '<tr>' + "\n".join(["<th>" + x + '</th>' for x in data[0]]) + '</tr>'
    for d in data:
        s += '<tr>' + "\n".join(["<td>" + str(d[x]) + '</td>' for x in d]) + '</tr>'
    s += '</table>'
    return s

def send_email(insert_data, update_data, mail_host, mail_user, mail_pass, sender, receivers):
    msg = ''
    if len(insert_data) > 0:
        msg += '<h1>insert</h1></br>' + table_html_generate(insert_data) + '</br></br>'
    if len(update_data) > 0:
        msg += '<h1>update</h1></br>' + table_html_generate(update_data) + '</br></br>'
    if msg == '':
        msg = '<h1>今日无新增数据</h1></br>'
        
    message = MIMEText(msg, 'html', 'utf-8')
    message['Subject'] = '牛客网{}招聘信息'.format(time.strftime("%Y-%m-%d"))
    message['From'] = sender
    message['To'] = receivers[0]
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)
        #smtpObj.connect(mail_host, 465)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(
            sender, receivers, message.as_string())
        smtpObj.quit()
        return True
    except smtplib.SMTPException as e:
        print('email send error: ', e)  
        return False    

def run(keywords, skip_words, db_config, mail_config = None):
    res = []
    for key in keywords:
        print(key, time.strftime("%Y-%m-%d %H:%M:%S"))
        for i in range(1, 11):
            print(i)
            a = get_newcoder_page(i, key)
            b = parse_newcoder_page(a, skip_words)
            if b == None or len(b) < 1: break
            res.extend(b)
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

def get_config(SERVER_ADDRESSES, NAMESPACE, GROUP):
    print(SERVER_ADDRESSES, NAMESPACE)
    client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE)

    keywords = json.loads(client.get_config("newcoder.crawler.keywords", GROUP))
    skip_words = json.loads(client.get_config("newcoder.crawler.skip_words", GROUP))
    db_config = json.loads(client.get_config("newcoder.crawler.db_config", GROUP))
    mail_config= json.loads(client.get_config("newcoder.crawler.mail_config", GROUP))
    return keywords, skip_words, db_config, mail_config

def main():
    SERVER_ADDRESSES = "ip:port"
    NAMESPACE = "your namespace"
    GROUP= "your group"
    
    run(*get_config(SERVER_ADDRESSES, NAMESPACE, GROUP))

if __name__ == "__main__":
    main()
    print("end")