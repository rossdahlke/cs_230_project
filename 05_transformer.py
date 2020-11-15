#####
# Ross Dahlke
# Identifying Opinion Change in Deliberative Settings
# Stanford University, CS 230: Deep Learning
#####

# This script contains the code for the final project
# Specifically, I'll be working on implementing transformers

# https://medium.com/analytics-vidhya/spam-ham-classification-using-lstm-in-pytorch-950daec94a7c
import docx2txt
import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split

# reading in the transcripts
deltas = pd.read_csv("data/processed/survey/opinion_deltas.csv")
doc_list = []
delta_list = []
for id in deltas["id"]:
    try:
        doc = docx2txt.process("data/processed/transcripts/" + id + ".docx").replace("\n", " ").lower()
        doc_list.append(doc)
        delta = abs(deltas[deltas["id"] == id]["delta"].values[0])
        delta_list.append(delta)
    except Exception:
        print("passed on " + id + ". No transcript.")
        pass

# split into train, test, and validation datasets
trn_idx, test_idx = train_test_split(np.arange(101), test_size = .1, random_state = 4)
trn_idx, val_idx = train_test_split(trn_idx, test_size = .1, random_state = 4)

# lets use some transformers
import torch
torch.cuda.empty_cache()

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

model.train()

from transformers import AdamW

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": .01},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr = 1e-5)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_batch = [doc_list[i] for i in trn_idx]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
labels = torch.tensor(np.round([delta_list[i] for i in trn_idx]))
labels = labels.type(torch.LongTensor)

test_batch = [doc_list[i] for i in test_idx]
test_encoding = tokenizer(test_batch, return_tensors='pt', padding=True, truncation=True)
test_input_ids = test_encoding['input_ids']
test_attention_mask = test_encoding["attention_mask"]
test_labels = torch.tensor(np.round([delta_list[i] for i in test_idx]))
test_labels = test_labels.type(torch.LongTensor)

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, InputFeatures

model = BertForSequenceClassification.from_pretrained("bert-large-uncased")

train_dataset = TensorDataset(input_ids, attention_mask, labels)

test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

def dummy_data_collector(features):
    batch = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[1] for f in features])
    batch['labels'] = torch.stack([f[2] for f in features])

    return batch

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=10,              # total # of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    data_collator = dummy_data_collector
)

trainer.train()

trainer.evaluate()

model(test_input_ids)
