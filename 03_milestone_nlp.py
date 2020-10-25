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
# 3. Baseline transformers

### tf-idf

## going to use spacy's vector similarity functionality
import spacy
import docx
import pandas as pd
import numpy as np

deltas = pandas.read_csv("data/processed/survey/opinion_deltas.csv")
similarity_list = []
delta_values = []
for id in deltas["id"]:
    print(id)
    similarities = []
    try:
        doc = docx.Document("data/processed/transcripts/" + id + ".docx")
        paragraphs = [p.text for p in doc.paragraphs][:-10]
        for i in range(len(paragraphs)-1):
            similarity = nlp(paragraphs[i]).similarity(nlp(paragraphs[i+1]))
            similarities.append(similarity)
        similarity_list.append(similarities)
        delta_values.append(deltas[deltas["id"] == id]["delta"].values[0])
    except Exception:
        print("passed on " + id + ". No transcript.")
        pass



## sentiment


## transformers
