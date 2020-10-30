#####
# Ross Dahlke
# Identifying Opinion Change in Deliberative Settings
# Stanford University, CS 230: Deep Learning
#####

# This script contains the code for the milestone
# For the milestone, I will be making a variety of baseline models
# These baseline models _will not_ include any linguistic features
# I will focus on linguistic features for the main model

# There will be three models that I will train
# 1. tf-idf with some sort of regression
# 2. RNN with pre-trained sentiment
# 3. LSTM (on text)
# 4. Various transformers

### tf-idf

## going to use spacy's vector similarity functionality to calculate the similarities of the word vectors for 1/10ths of each transcript
import spacy
nlp = spacy.load("en_core_web_lg")
import docx
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats

## going to try more similarity stuff
import itertools
from sklearn.model_selection import train_test_split

def divide_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns = data_dict.keys())

deltas = pd.read_csv("data/processed/survey/opinion_deltas.csv")
deltas_similarities = pd.DataFrame()
for id in deltas["id"]:
    n_chunks = 10
    try:
        doc = docx.Document("data/processed/transcripts/" + id + ".docx")
        paragraphs = [nlp(p.text) for p in doc.paragraphs]
        chunks = list(divide_chunks(paragraphs, n_chunks))
        expanded_grid = expand_grid({"text_0": range(0, 10),
                                    "text_1": range(0, 10)})

        similarities = []
        for i in range(0, len(expanded_grid)):
            similarities.append(paragraphs[expanded_grid["text_0"][i]].similarity(paragraphs[expanded_grid["text_1"][i]]))
        expanded_grid["similarity"] = similarities
        expanded_grid["comparison"] = expanded_grid["text_0"].astype(str) + "_" + expanded_grid["text_1"].astype(str)
        expanded_grid["delta"] = abs(deltas[deltas["id"] == id]["delta"].values[0])
        expanded_grid_wide = expanded_grid.pivot(index = "delta", columns = "comparison", values = "similarity").reset_index()
        deltas_similarities = deltas_similarities.append(expanded_grid_wide)
        print()
        print(id)

    except Exception:
        print()
        print("passed on " + id + ". No transcript.")
        pass

deltas_similarities_dropped = deltas_similarities[['delta', '0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7', '0_8', '0_9', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7', '1_8', '1_9', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '4_5', '4_6', '4_7', '4_8', '4_9', '5_6', '5_7', '5_8', '5_9', '6_7', '6_8', '6_9', '7_8', '7_9', '8_9']]

## linear regression
trn_idx, test_idx = train_test_split(np.arange(101), test_size = .1, random_state = 1)

model = LinearRegression()

# all similarities
X = deltas_similarities_dropped[['0_1', '0_2', '0_3', '0_4', '0_5', '0_6', '0_7', '0_8', '0_9', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7', '1_8', '1_9', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '4_5', '4_6', '4_7', '4_8', '4_9', '5_6', '5_7', '5_8', '5_9', '6_7', '6_8', '6_9', '7_8', '7_9', '8_9']]

model.fit(X.iloc[trn_idx], deltas_similarities_dropped[["delta"]].iloc[trn_idx])

test_pred = model.predict(X.iloc[test_idx])

mean_squared_error(deltas_similarities_dropped[["delta"]].iloc[test_idx], test_pred)

r2_score(deltas_similarities_dropped[["delta"]].iloc[test_idx], test_pred)

# Just 0_9

X = deltas_similarities_dropped[['1_9']]

model.fit(X.iloc[trn_idx], deltas_similarities_dropped[["delta"]].iloc[trn_idx])

test_pred = model.predict(X.iloc[test_idx])

mean_squared_error(deltas_similarities_dropped[["delta"]].iloc[test_idx], test_pred)

r2_score(deltas_similarities_dropped[["delta"]].iloc[test_idx], test_pred)

## going to try to fit a basic NN
# https://www.kaggle.com/aakashns/pytorch-basics-linear-regression-from-scratch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

targets = np.reshape(np.array(deltas_similarities_dropped["delta"]), (-1, 1))
inputs = deltas_similarities_dropped.drop(columns = "delta").values

# The rest of data will be test set
data_length = len(targets)
idx_array = np.array(list(range(0, data_length)))
np.random.shuffle(idx_array)
train_idx = idx_array[:int(data_length * .7)]
val_idx = idx_array[int(data_length * .7):int(data_length * .9)]
test_idx = idx_array[int(data_length * .9):]

train_inputs = torch.tensor(inputs[train_idx])
train_targets = torch.tensor(targets[train_idx])
val_inputs = torch.tensor(inputs[val_idx])
val_targets = torch.tensor(targets[val_idx])
test_inputs = torch.tensor(inputs[test_idx])
test_targets = torch.tensor(targets[test_idx])

train_ds = TensorDataset(train_inputs.float(), train_targets.float())
val_ds = TensorDataset(val_inputs.float(), val_targets.float())
test_ds = TensorDataset(test_inputs.float(), test_targets.float())
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size, shuffle=True)

class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(45, 10)
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(10, 1)

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x

model = SimpleNet()

opt = torch.optim.SGD(model.parameters(), lr=1e-5)
criterion = F.mse_loss

def fit(num_epochs, model, loss_fn, opt, print_every):
    counter = 0
    for epoch in range(num_epochs):
        counter += 1
        train_losses_in_itr = []
        for xb, yb in train_dl:
            # Generate predictions
            pred = model(xb.float())
            loss = criterion(pred.float(), yb.float())
            train_losses_in_itr.append(loss.item())
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss

            val_losses_in_itr = []

            for xb, yb in val_dl:
                # Generate predictions
                pred = model(xb.float())
                val_loss = criterion(pred.float(), yb.float())
                val_losses_in_itr.append(val_loss.item())

            print("Epoch: {:2d}/{:2d}\t".format(epoch+1, num_epochs),
                  "Train Loss: {:.6f}\t".format(np.mean(train_losses_in_itr)),
                  "Val Loss: {:.6f}\t".format(np.mean(val_losses_in_itr)))

fit(3000, model, criterion, opt, print_every = 100)

criterion(model(test_inputs.float()), test_targets.float())
