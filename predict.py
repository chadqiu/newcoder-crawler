from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from datasets import Dataset, DatasetDict

data = pd.read_excel("historical_data.xlsx", sheet_name=0).fillna(" ")

raw_datasets = DatasetDict()
raw_datasets['test'] = Dataset.from_pandas(data)

model_name = "/root/autodl-tmp/roberta"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda:0"
model.half()
model = model.to(device)

s = "秋招大结局（泪目了）。家人们泪目了，一波三折之后获得的小奖状，已经准备春招了，没想到被捞啦，嗐，总之是有个结果，还是很开心的[掉小珍珠了][掉小珍珠了]"

max_target_length = 512
label2id = {
    '招聘信息':0,
    '经验贴':1,
    '求助贴':2
}
id2label = {v:k for k,v in label2id.items()}


def get_answer(text):
    text = [x for x in text]
    inputs = tokenizer( text, return_tensors="pt", max_length=max_target_length, padding=True, truncation=True)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    # print(inputs)
    with torch.no_grad():
        outputs = model(**inputs).logits.argmax(-1).tolist()
    return outputs

data['text'] = data['title'].apply(str) + data['content'].apply(str)

# print(get_answer(data['text'][:10]))

pred , grod = [], []
index, batch_size = 0, 32

while index < len(data['text']):
    pred.extend(get_answer([x for x in data['text'][index:index + batch_size]]))
    index += batch_size
    # print(pred[-1], grod[-1])

# print(pred)
# print(grod)

pred = [id2label[x] for x in pred]
data["target"] = pred

writer = pd.ExcelWriter("generate.xlsx")
data.to_excel(writer, index=False, encoding='utf-8', sheet_name='Sheet1')
writer.save()
writer.close()

# num = 0
# for x,y in zip(pred,grod):
#     if x != y:
#         num += 1
# print(num, num / len(pred))


