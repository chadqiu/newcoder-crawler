# newcoder-crawler
爬虫牛客网帖子，获取感兴趣信息，每天增量更新，获取最新数据，发邮件提醒，使用MySQL保存历史数据。使用Nacos配置中心动态改变关键词、接收邮箱等信息。

详细介绍见博客 [牛客网爬虫](https://chadqiu.github.io/f06a19b2ce94.html)

附上爬取的历史数据和数据库表的定义信息，用于后续的模型训练

crawler.py 为最基本的爬虫代码，可以实现指定关键词对牛客网帖子的爬取，主要是工作信息，使用简单的关键词对无关文档进行过滤，数据每日更新并发送邮件通知，详细内容见博客 [牛客网爬虫](https://chadqiu.github.io/f06a19b2ce94.html)

由于过滤的关键词，邮件接收列表经常变化，修改代码再重启很麻烦，引入了配置中心nacos，通过网页更改配置，代码会自动获取最新的配置信息，而无需改代码重启项目。更改后的代码在 crawler_registry中。

基于关键词对无关帖子进行过滤，会有很多误差，最好使训练一个NLP分类模型进行过滤，本人从零开始构建了一个网络讨论帖分类模型，用其对无关类容进行过滤，更改后的代码在 crawler_advanced.py中。 roberta4h512.zip便是最终训练好的模型，可用其进行推理过滤，使用时将其解压到与crawler_advanced.py同级目录， 搭建模型的详细过程见博客[如何从零开始构建一个网络讨论帖分类模型？](https://chadqiu.github.io/ed5507eb2665.html)。

**文件说明**：

使用bert，roberta等encoder only 模型，bert_train.py 为训练代码， predict为预测代码 
使用T5等 encoder - decoder 模型， seq2seq.py是训练代码， generate.py为预测代码

train.csv 为使用chatGPT标注生成的数据，
test.csv为人工标注的数据。
historical_data.py为从牛客网爬取的近4万条无标签数据，
generated_pesudo_data.csv为使用训练后的uer roberta-large模型标注生成的数据。
roberta4h512.zip保存的是最终用来部署的roberta-small模型，使用时将其解压到当前代码所在的目录，其在test.csv测试集的表现如下：

![f1 score](https://chadqiu.github.io/images/newcoder_f1.png)
