#####
# Ross Dahlke
# Identifying Opinion Change in Deliberative Settings
# Stanford University, CS 230: Deep Learning
#####

# This file does all of the data processing for the training data
# I start with "raw" data in the format that I recieved from the Center for Deliberative Democracy
# This "raw" data is stored in "data/raw"
# And the processed data is stored in "data/processed"
# Since there are only 3 datasets, and each is formatting uniquely, I will work with each dataset individually instead of functionalizing
# Funcationalization could be done if more Deliberative Polls are conducted and data is publicly released

# There are two main processing tasks:
# 1. Processing survey results to calcualte deltas in opinions
# 2. Text pre-processing on the transcripts

# There are multiple different Deliberative Polls, each with about 15 groups which conduct 2-3 different sessions
# To standardize the data, I will labels transcripts and survey results as:
# dp[a]_group[b]_session[1]
# For example, Deliberative Poll #1, Group #2, Session #3 will be labeled as:
# dp1_group2_session3

# start by importing packages
import pandas as pd
import numpy as np

### 1. Processing survey results to calculate deltas in opinions

## First, let's read in the Ghana transcripts
ghana_pre = pd.read_csv("data/raw/surveys/ghana-presurvey.csv")
ghana_post = pd.read_csv("data/raw/surveys/ghana-postsurvey.csv")

# read in the ghana groups
ghana_groups = pd.read_csv("data/raw/surveys/ghana_groups.csv")

# merging with group numbers
ghana_pre = ghana_pre.merge(ghana_groups[["formnumber", "groupnumber"]], how = "left", on = "formnumber")

# this join key isn't perfect and some people are't getting a group
# if I have time I can come back to try to find a combination of columns that could be used
ghana_post = ghana_post.merge(ghana_groups[["namerespon", "groupnumber"]], how = "left", on = "namerespon")

# cleaning up the questions I will ultimately use
# don't have good data on movement since I don't know group number
# I'll just use all topic related questions until I get that information
ghana_pre[["q3", "q12", "q14", "q41d", "q5", "q20", "q29", "q33"]] = ghana_pre[["q3", "q12", "q14", "q41d", "q5", "q20", "q29", "q33"]].replace(["Extremely important", "Exactly in Middle", "Extremely Unimportant", "Don't know"], [10, 5, 0, None]).replace(99, 5)
ghana_post[["q3", "q12", "q14", "q41d", "q5", "q20", "q29", "q33"]] = ghana_post[["q3", "q12", "q14", "q41d", "q5", "q20", "q29", "q33"]].replace(["Extremely important", "Exactly in Middle", "Extremely Unimportant", "Don't know"], [10, 5, 0, None]).replace(99, 5)

# food safety/ livelihood
ghana_pre["qfood"] = (pd.to_numeric(ghana_pre["q3"]) + pd.to_numeric(ghana_pre["q12"]) + pd.to_numeric(ghana_pre["q14"])) / 3
ghana_pre_qfood = ghana_pre.groupby("group", as_index = False)["qfood"].mean()

ghana_post["qfood"] = (pd.to_numeric(ghana_post["q3"]) + pd.to_numeric(ghana_post["q12"]) + pd.to_numeric(ghana_post["q14"])) / 3
ghana_post_qfood = ghana_post.groupby("group", as_index = False)["qfood"].mean()

ghana_qfood_delta = pd.DataFrame({"group": ghana_post_qfood["group"], "delta": ghana_post_qfood["qfood"] - ghana_pre_qfood["qfood"]})
ghana_qfood_delta["id"] = "dp1_group" + ghana_qfood_delta["group"].astype(str) + "_session1"
ghana_qfood_delta = ghana_qfood_delta.drop(["group"], axis = 1)

# water policy
ghana_pre["qwater"] = (pd.to_numeric(ghana_pre["q5"]) + pd.to_numeric(ghana_pre["q20"]) + pd.to_numeric(ghana_pre["q29"]) + pd.to_numeric(ghana_pre["q33"])) / 4
ghana_pre_qwater = ghana_pre.groupby("group", as_index = False)["qwater"].mean()

ghana_post["qwater"] = (pd.to_numeric(ghana_post["q5"]) + pd.to_numeric(ghana_post["q20"]) + pd.to_numeric(ghana_post["q29"]) + pd.to_numeric(ghana_post["q33"])) / 4
ghana_post_qwater = ghana_post.groupby("group", as_index = False)["qwater"].mean()

ghana_qwater_delta = pd.DataFrame({"group": ghana_post_qwater["group"], "delta": ghana_post_qwater["qwater"] - ghana_pre_qwater["qwater"]})
ghana_qwater_delta["id"] = "dp1_group" + ghana_qwater_delta["group"].astype(str) + "_session2"
ghana_qwater_delta = ghana_qwater_delta.drop(["group"], axis = 1)

## Onto the Bududa
bududa_and_butaleja = pd.read_excel("data/raw/surveys/uganda_Deliberative polling_Pre& Post Survey Data.xlsx")

# going to add that random group for now, same as Ghana
bududa_pre = bududa_and_butaleja[((bududa_and_butaleja["000ID"].str.contains("BUD")) & (bududa_and_butaleja["001_Poll"] == 1))]
bududa_post = bududa_and_butaleja[((bududa_and_butaleja["000ID"].str.contains("BUD")) & (bududa_and_butaleja["001_Poll"] == 2))]

bududa_pre["group"] = np.random.randint(1, 13, bududa_pre.shape[0])
bududa_pre["group"] = bududa_pre["group"].replace([12], [14])

bududa_post["group"] = np.random.randint(1, 13, bududa_post.shape[0])
bududa_post["group"] = bududa_post["group"].replace([12], [14])
