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

sim_matrix_ols = ols('delta ~ Q("0_1") + Q("0_2") + Q("0_3") + Q("0_4") + Q("0_5") + Q("0_6") + Q("0_7") + Q("0_8") + Q("0_9") + Q("1_2") + Q("1_3") + Q("1_4") + Q("1_5") + Q("1_6") + Q("1_7") + Q("1_8") + Q("1_9") + Q("2_3") + Q("2_4") + Q("2_5") + Q("2_6") + Q("2_7") + Q("2_8") + Q("2_9") + Q("3_4") + Q("3_5") + Q("3_6") + Q("3_7") + Q("3_8") + Q("3_9") + Q("4_5") + Q("4_6") + Q("4_7") + Q("4_8") + Q("4_9") + Q("5_6") + Q("5_7") + Q("5_8") + Q("5_9") + Q("6_7") + Q("6_8") + Q("6_9") + Q("7_8") + Q("7_9") + Q("8_9")', deltas_similarities_dropped)

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

## going to try to fit a DL model
