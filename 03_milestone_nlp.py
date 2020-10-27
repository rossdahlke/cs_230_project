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

## going to use spacy's vector similarity functionality
import spacy
nlp = spacy.load("en_core_web_lg")
import docx
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats

# similarities
deltas = pd.read_csv("data/processed/survey/opinion_deltas.csv")
similarity_list = []
delta_list = []
for id in deltas["id"]:
    n_breaks = 10
    try:
        doc = docx.Document("data/processed/transcripts/" + id + ".docx")
        paragraphs = [p.text for p in doc.paragraphs][:-10]
        break_index = round(len(paragraphs) / n_breaks)
        s = " "
        text_0 = nlp(s.join(paragraphs[:break_index]))
        text_1 = nlp(s.join(paragraphs[-break_index * (n_breaks - 1):]))
        similarity = text_0.similarity(text_1)
        similarity_list.append(similarity)
        delta = abs(deltas[deltas["id"] == id]["delta"].values[0])
        delta_list.append(delta)
        print()
        print(id)
        print(similarity)
        print(delta)
    except Exception:
        print("passed on " + id + ". No transcript.")
        pass

# linear regression
similarity_train = np.reshape(np.asarray(similarity_list[0:90]), (-1, 1))
similarity_test = np.reshape(np.asarray(similarity_list[91:101]), (-1, 1))

delta_train = np.reshape(np.asarray(delta_list[0:90]), (-1, 1))
delta_test = np.reshape(np.asarray(delta_list[91:101]), (-1, 1))

model = LinearRegression()

model.fit(similarity_train, delta_train)

delta_test_pred = model.predict(similarity_test)

model.coef_

mean_squared_error(delta_test, delta_test_pred)

r2_score(delta_test, delta_test_pred)

plt.scatter(similarity_test, delta_test, color = "black")
plt.plot(similarity_test, delta_test_pred, color = "blue", linewidth = 3)
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(similarity_list, delta_list)

slope
p_value

## RNN with sentiment
# sentiment analysis based off: https://medium.com/@b.terryjack/nlp-pre-trained-sentiment-analysis-1eb52a9d742c
from textblob import TextBlob

polarity_list = []
subjectivity_list = []
delta_list = []
for id in deltas["id"]:
    polarities = []
    subjectivities = []
    try:
        doc = docx.Document("data/processed/transcripts/" + id + ".docx")
        paragraphs = [p.text for p in doc.paragraphs][:-10]
        for paragraph in paragraphs:
            polarity = TextBlob(paragraph).sentiment[0]
            polarities.append(polarity)
            subjectivity = TextBlob(paragraph).sentiment[1]
            subjectivities.append(subjectivity)
        polarity_list.append(polarities)
        subjectivity_list.append(subjectivities)
        delta = abs(deltas[deltas["id"] == id]["delta"].values[0])
        delta_list.append(delta)
    except Exception:
        print("passed on " + id + ". No transcript.")
        pass

# RNN https://www.kaggle.com/purplejester/pytorch-deep-time-series-classification
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader

seed = 1
np.random.seed(seed)

# can do some preprocessing in the future, but it shouldn't be too bad given range from the sentiment models

polarity_list_padded = []
for list in polarity_list:
    to_pad = max - len(list)
    polarity_list_padded.append(np.pad(list, (to_pad, 0)))
polarity_array = np.asarray(polarity_list_padded)

subjectivity_list_padded = []
for list in subjectivity_list:
    to_pad = max - len(list)
    subjectivity_list_padded.append(np.pad(list, (to_pad, 0)))
subjectivity_array = np.asarray(subjectivity_list_padded)

delta_array = np.asarray(delta_list)

def create_datasets(data, target, train_size, valid_pct=0.1, seed=None):
    """Converts NumPy arrays into PyTorch datsets.

    Three datasets are created in total:
        * training dataset
        * validation dataset
        * testing (un-labelled) dataset

    """
    sz = train_size
    idx = np.arange(sz)
    trn_idx, val_idx = train_test_split(
        idx, test_size=valid_pct, random_state=seed)
    trn_ds = TensorDataset(
        torch.tensor(data[:sz][trn_idx]),
        torch.tensor(target[:sz][trn_idx]).long())
    val_ds = TensorDataset(
        torch.tensor(data[:sz][val_idx]),
        torch.tensor(target[:sz][val_idx]).long())
    tst_ds = TensorDataset(
        torch.tensor(data[sz:]),
        torch.tensor(target[sz:]).long())
    return trn_ds, val_ds, tst_ds












## LSTM on text
