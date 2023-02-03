import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from transformers import EvalPrediction

import evaluate

metric = evaluate.load("seqeval")

#  hfl/rbt3
#  uer/chinese_roberta_L-4_H-512
#  uer/roberta-large-wwm-chinese-cluecorpussmall
model_name = "uer/chinese_roberta_L-4_H-512"
# model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

max_input_length = 128
max_target_length = 20
label2id = {
    '招聘信息':0,
    '经验贴':1,
    '求助贴':2
}
id2label = {v:k for k,v in label2id.items()}

def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=max_input_length, truncation=True)
    labels = [label2id[x] for x in examples['target']]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)


def multi_label_metrics(predictions, labels, threshold=0.5):
    probs =  np.argmax( predictions, -1)       
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=probs, average='micro')
    accuracy = accuracy_score(y_true, probs)
    print(classification_report([id2label[x] for x in y_true], [id2label[x] for x in probs]))
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'accuracy': accuracy}
    return metrics
 
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                        # problem_type="multi_label_classification", 
                                        num_labels=3,
                                        # id2label=id2label,
                                        # label2id=label2id
                                        )

batch_size = 64
metric_name = "f1"

training_args = TrainingArguments(
    f"/root/autodl-tmp/run",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # gradient_accumulation_steps=2,
    num_train_epochs=10,
    save_total_limit=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    fp16=True,
)




trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


print("test")
print(trainer.evaluate())
# trainer.save_model("bert")

predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=-1)

print(predictions)
print(labels)

