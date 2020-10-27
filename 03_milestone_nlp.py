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

deltas = pd.read_csv("data/processed/survey/opinion_deltas.csv")
similarity_list = []
delta_list = []
id = deltas["id"][0]
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



## baseline NN
# using this tutorial https://www.kaggle.com/purplejester/a-simple-lstm-based-time-series-classifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

similarity_tensors = []
for i in range(len(similarity_list)):
    similarity_tensors.append(torch.tensor(similarity_list[i]))

similarity_tensors_padded = pad_sequence(similarity_tensors)
