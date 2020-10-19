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

# going to add a random column with a group number since I don't have it yet, will remove once I get it
ghana_pre["group"] = np.random.randint(1, 13, ghana_pre.shape[0])
ghana_pre["group"] = ghana_pre["group"].replace([4], [15])

ghana_post["group"] = np.random.randint(1, 13, ghana_post.shape[0])
ghana_post["group"] = ghana_post["group"].replace([4], [15])

# unfortunately I do not have a key for know what the survey questions asked, I'll just use q1 and q2 as the responses to the issues discussed in the sessions
# have to do a little work to clean up responses
ghana_pre[["q1", "q2"]] = ghana_pre[["q1", "q2"]].replace(["Extremely important", "Exactly in Middle", "Extremely Unimportant", "Don't know"], [10, 5, 0, None])
ghana_post[["q1", "q2"]] = ghana_post[["q1", "q2"]].replace(["Extremely important", "Exactly in Middle", "Extremely Unimportant", "Don't know"], [10, 5, 0, None])

# q1
ghana_pre_q1 = ghana_pre[(ghana_pre["q1"] != "NA")]
ghana_pre_q1["q1"] = pd.to_numeric(ghana_pre_q1["q1"])
ghana_pre_q1_summarized = ghana_pre_q1.groupby("group", as_index = False)["q1"].mean()

ghana_post_q1 = ghana_post[(ghana_post["q1"] != "NA")]
ghana_post_q1["q1"] = pd.to_numeric(ghana_post_q1["q1"])
ghana_post_q1_summarized = ghana_post_q1.groupby("group", as_index = False)["q1"].mean()

ghana_q1_delta = pd.DataFrame({"group": ghana_post_q1_summarized["group"], "delta": ghana_post_q1_summarized["q1"] - ghana_pre_q1_summarized["q1"]})
ghana_q1_delta["id"] = "dp1_group" + ghana_q1_delta["group"].astype(str) + "_session1"
ghana_q1_delta = ghana_q1_delta.drop(["group"], axis = 1)

# q2
ghana_pre_q2 = ghana_pre[(ghana_pre["q2"] != "NA")]
ghana_pre_q2["q2"] = pd.to_numeric(ghana_pre_q2["q2"])
ghana_pre_q2_summarized = ghana_pre_q2.groupby("group", as_index = False)["q2"].mean()

ghana_post_q2 = ghana_post[(ghana_post["q2"] != "NA")]
ghana_post_q2["q2"] = pd.to_numeric(ghana_post_q1["q2"])
ghana_post_q2_summarized = ghana_post_q2.groupby("group", as_index = False)["q2"].mean()

ghana_q2_delta = pd.DataFrame({"group": ghana_post_q2_summarized["group"], "delta": ghana_post_q2_summarized["q2"] - ghana_pre_q2_summarized["q2"]})
ghana_q2_delta["id"] = "dp1_group" + ghana_q2_delta["group"].astype(str) + "_session1"
ghana_q2_delta = ghana_q2_delta.drop(["group"], axis = 1)
