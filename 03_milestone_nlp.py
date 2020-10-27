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



## LSTM on text
