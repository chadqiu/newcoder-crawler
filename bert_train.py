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
model_name = "/root/autodl-tmp/l3h512"
# model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

raw_datasets = load_dataset('csv', data_files={'train': 'generate_l3.csv', 'test': 'test.csv'})

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
    # gradient_checkpointing=True,
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
    #push_to_hub=True,
)


import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

# training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)

decay_parameters = get_parameter_names(model, [nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": training_args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer_kwargs = {
    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    "eps": training_args.adam_epsilon,
}
optimizer_kwargs["lr"] = training_args.learning_rate
adam_bnb_optim = bnb.optim.Adam8bit(
    optimizer_grouped_parameters,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    lr=training_args.learning_rate,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(adam_bnb_optim, None)
)

# trainer.train()


# from accelerate import Accelerator
# from torch.utils.data.dataloader import DataLoader
# tokenized_datasets['train'].set_format("pt")
# dataloader = DataLoader(tokenized_datasets['train'], batch_size=training_args.per_device_train_batch_size)

# if training_args.gradient_checkpointing:
#     model.gradient_checkpointing_enable()

# accelerator = Accelerator(fp16=training_args.fp16)
# model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

# model.train()
# for i in range(training_args.num_train_epochs):
#     for step, batch in enumerate(dataloader, start=1):
#         loss = model(**batch).loss
#         loss = loss / training_args.gradient_accumulation_steps
#         accelerator.backward(loss)
#         if step % training_args.gradient_accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()
  

# model = AutoModelForMaskedLM.from_pretrained("/root/autodl-tmp/Deberta")
print("test")
print(trainer.evaluate())
# trainer.save_model("bert")

predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=-1)

print(predictions)
print(labels)

#  uer  64  914  933
#  uer2  64  914
#  l3 64 90