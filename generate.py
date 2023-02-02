# 使用文本生成的方式进行文本分类， 最终未采用

from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import pandas as pd

data = pd.read_excel("historical_data.xlsx", sheet_name = 0).fillna(" ")

device = "cuda:0"

model_name = "model"
max_target_length = 512
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.half()
model = model.to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)
prefix = "请问下面文本属于 招聘信息、 经验贴、 求助贴 三者中的哪一类？\n"
suffix = "\n选项：招聘信息, 经验贴, 求助贴\n答案："


def get_answer(text):
    if not text :
        return "null"
    inputs = tokenizer( prefix + str(text) + suffix, return_tensors="pt", max_length=max_target_length, truncation=True)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    # print(inputs)
    outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True)
    return tokenizer.decode(outputs[0][0], skip_special_tokens=True)


data['text'] = data['title'].apply(str) + data['content'].apply(str)
data['target'] = data['text'].map(get_answer)  # not recommend, it's better to generate in batches 

writer = pd.ExcelWriter("generate.xlsx")
data.to_excel(writer, index=False, encoding='utf-8', sheet_name='Sheet1')
writer.save()
writer.close()


def cal_accuracy():
    pred , grod = [], []

    for t, l in zip(data['text'], data['target']):
        pred.append(get_answer(t))
        grod.append(l)
        # print(pred[-1], grod[-1])
    print(pred)
    print(grod)

    num = 0
    for x,y in zip(pred,grod):
        if x != y:
            num += 1
    print(num, 1 - num / len(pred))


