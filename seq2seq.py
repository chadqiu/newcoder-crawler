import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import evaluate

metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = [tokenizer.batch_decode(predictions, skip_special_tokens=True)] 
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = [tokenizer.batch_decode(labels, skip_special_tokens=True)] 
    return metric.compute(predictions=decoded_preds, references=decoded_labels)




model_name = "ClueAI/ChatYuan-large-v1"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
# metric = load_metric("rouge")

raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

max_input_length = 252
max_target_length = 20
prefix = "请问下面文本属于 招聘信息、 经验贴、 求助贴 三者中的哪一类？\n"
suffix = "\n选项：招聘信息, 经验贴, 求助贴\n答案："

def preprocess_function(examples):
    inputs = [prefix + doc + suffix for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

batch_size = 4

args = Seq2SeqTrainingArguments(
    f"yuan-finetuned-xsum",
    evaluation_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 10,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    # fp16=True,
    # push_to_hub=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

model = AutoModelForSeq2SeqLM.from_pretrained("model")
# trainer.train()
print("test")
print(trainer.evaluate())
# trainer.save_model("yun")

# maxc = 0
# result = []
# for i in range(10):
#     trainer.train()
#     a = trainer.evaluate()
#     result.append(a)
#     print(a)
#     if a['eval_overall_accuracy'] > maxc:
#         maxc = a['eval_overall_accuracy']
#         trainer.save_model("yuan")

# print(result)