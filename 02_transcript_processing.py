#####
# Ross Dahlke
# Identifying Opinion Change in Deliberative Settings
# Stanford University, CS 230: Deep Learning
#####

# This file does all of the data processing for the training data
# I start with "raw" data in the format that I recieved from the Center for Deliberative Democracy
# This "raw" data is stored in "data/raw"
# And the processed data is stored in "data/processed"

# The main task for this script:
# 1. Rename transcripts so that they are easy read in during model building

# important: for now, this script will only rename files to fit the naming convention below
# since there are a variety of NLP tasks that I will be using these transcripts for,
# different preprocessing is needed for different tasks
# depending on how long preprocessing takes, I will either just include it in the analysis script
# or I will append preprocessing onto this script

# There are multiple different Deliberative Polls, each with about 15 groups which conduct 2-3 different sessions
# To standardize the data, I will labels transcripts and survey results as:
# dp[a]_group[b]_session[1]
# For example, Deliberative Poll #1, Group #2, Session #3 will be labeled as:
# dp1_group2_session3

# start by importing packages
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# let's get a list of files we want to read in
transcripts = [f for f in listdir("data/raw/transcripts") if isfile(join("data/raw/transcripts", f))]

old_file = transcripts[25]

def rename_transcript(old_file):
# params: x = name of transcript
# returns: new_file = name of new standardized file name
# DP number
    if "ghana" in old_file:
        dp_n = "1"
    if "uganda" in old_file:
        dp_n = "2"
    if "butaleja" in old_file:
        dp_n = "3"

    # group number
    if "ghana" in old_file:
        group_n = old_file[old_file.find("GRP") + len("GRP"):old_file.rfind(".docx")]
    if "uganda" in old_file:
        group_n = old_file.split("_")[1].replace("group", "")
    if "butaleja" in old_file:
        group_n = old_file.split("_")[1].replace("group", "")

    # session number
    if "ghana" in old_file:
        session_n = [int(s) for s in old_file if s.isdigit()][0]
    if "uganda" in old_file:
        if "land" in old_file.lower():
            session_n = 1
        if "population" in old_file.lower():
            session_n = 2
        if "resettlement" in old_file.lower():
            session_n = 3
    if "butaleja" in old_file:
        if "land" in old_file.lower():
            session_n = 1
        if "population" in old_file.lower():
            session_n = 2
        if "resettlement" in old_file.lower():
            session_n = 3

    new_file = "dp" + str(dp_n) + "_group" + str(group_n) + "_session" + str(session_n) + ".docx"

    return new_file

for x in transcripts:
    new_file = rename_transcript(x)
    os.rename(r"data/raw/transcripts/"+x, r"data/processed/transcripts/"+new_file)
