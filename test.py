from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import pandas as pd

data = pd.read_csv("test.csv")

model_name = "model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
prefix = "请问下面文本属于 招聘信息、 经验贴、 求助贴 三者中的哪一类？\n"
suffix = "\n选项：招聘信息, 经验贴, 求助贴\n答案："


def get_answer(text):
    inputs = tokenizer( prefix + text + suffix, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True)
    return tokenizer.decode(outputs[0][0], skip_special_tokens=True)

pred , grod = [], []

for t, l in zip(data['text'], data['target']):
    pred.append(get_answer(t))
    grod.append(l)

print(pred)
print(grod)