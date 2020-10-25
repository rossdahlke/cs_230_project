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

# Main task for this document:
# 1. Processing survey results to calcualte deltas in opinions

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
# replace "99" (no response) with 5s (middle of the likert scale)
# could get fancier in the future with interpolation in the future
ghana_pre[["q3", "q12", "q14", "q41d", "q5", "q20", "q29", "q33"]] = ghana_pre[["q3", "q12", "q14", "q41d", "q5", "q20", "q29", "q33"]].replace(["Extremely important", "Exactly in Middle", "Extremely Unimportant", "Don't know"], [10, 5, 0, None]).replace(99, 5)
ghana_post[["q3", "q12", "q14", "q41d", "q5", "q20", "q29", "q33"]] = ghana_post[["q3", "q12", "q14", "q41d", "q5", "q20", "q29", "q33"]].replace(["Extremely important", "Exactly in Middle", "Extremely Unimportant", "Don't know"], [10, 5, 0, None]).replace(99, 5)

# food safety/ livelihood
ghana_pre["qfood"] = (pd.to_numeric(ghana_pre["q3"]) + pd.to_numeric(ghana_pre["q12"]) + pd.to_numeric(ghana_pre["q14"])) / 3
ghana_pre_qfood = ghana_pre.groupby("groupnumber", as_index = False)["qfood"].mean()

ghana_post["qfood"] = (pd.to_numeric(ghana_post["q3"]) + pd.to_numeric(ghana_post["q12"]) + pd.to_numeric(ghana_post["q14"])) / 3
ghana_post_qfood = ghana_post.groupby("groupnumber", as_index = False)["qfood"].mean()

ghana_qfood_delta = pd.DataFrame({"group": ghana_post_qfood["groupnumber"].astype(int).astype(str), "delta": ghana_post_qfood["qfood"] - ghana_pre_qfood["qfood"]})
ghana_qfood_delta["id"] = "dp1_group" + ghana_qfood_delta["group"].astype(str) + "_session1"
ghana_qfood_delta = ghana_qfood_delta.drop(["group"], axis = 1)

# water policy
ghana_pre["qwater"] = (pd.to_numeric(ghana_pre["q5"]) + pd.to_numeric(ghana_pre["q20"]) + pd.to_numeric(ghana_pre["q29"]) + pd.to_numeric(ghana_pre["q33"])) / 4
ghana_pre_qwater = ghana_pre.groupby("groupnumber", as_index = False)["qwater"].mean()

ghana_post["qwater"] = (pd.to_numeric(ghana_post["q5"]) + pd.to_numeric(ghana_post["q20"]) + pd.to_numeric(ghana_post["q29"]) + pd.to_numeric(ghana_post["q33"])) / 4
ghana_post_qwater = ghana_post.groupby("groupnumber", as_index = False)["qwater"].mean()

ghana_qwater_delta = pd.DataFrame({"group": ghana_post_qwater["groupnumber"].astype(int).astype(str), "delta": ghana_post_qwater["qwater"] - ghana_pre_qwater["qwater"]})
ghana_qwater_delta["id"] = "dp1_group" + ghana_qwater_delta["group"].astype(str) + "_session2"
ghana_qwater_delta = ghana_qwater_delta.drop(["group"], axis = 1)

## Onto the Bududa
# also going to replace "99" (no response) with 5 (middle of the likert scale)
# could do interpolation in the future
bududa_and_butaleja = pd.read_excel("data/raw/surveys/uganda_Deliberative polling_Pre& Post Survey Data.xlsx").replace(99, 5)

# getting bududa groups
bududa_groups = pd.read_csv("data/raw/surveys/bududa_groups.csv")

# filtering to just the Bududa data, also joining on groups
bududa_pre = bududa_and_butaleja[((bududa_and_butaleja["000ID"].str.contains("BUD")) & (bududa_and_butaleja["001_Poll"] == 1))].merge(bududa_groups, how = "left", left_on = "000ID", right_on = "id")
bududa_post = bududa_and_butaleja[((bududa_and_butaleja["000ID"].str.contains("BUD")) & (bududa_and_butaleja["001_Poll"] == 2))].merge(bududa_groups, how = "left", left_on = "000ID", right_on = "id")

# Land Management
bududa_pre["land_management"] = bududa_pre[["114_Planttrees_protectriverbeds", "115_Riverchannels_localgovernment", "116_wetlands_dryseason", "117_Riceschemes_notinwetlands", "118_Communities_maintainwaterchannels", "119_Communities_benefitscropdiversity", "120_Communities_de-silting", "121_Government_assistdesilting", "122_Communities_sanitationdrains", "123_Government_drillingcleanwater", "124_Communities_resourcesaccesswater"]].sum(axis = 1) / 11

bududa_post["land_management"] = bududa_post[["114_Planttrees_protectriverbeds", "115_Riverchannels_localgovernment", "116_wetlands_dryseason", "117_Riceschemes_notinwetlands", "118_Communities_maintainwaterchannels", "119_Communities_benefitscropdiversity", "120_Communities_de-silting", "121_Government_assistdesilting", "122_Communities_sanitationdrains", "123_Government_drillingcleanwater", "124_Communities_resourcesaccesswater"]].sum(axis = 1) / 11

bududa_pre_lm = bududa_pre.groupby("groupnumber", as_index = False)["land_management"].mean()
bududa_post_lm = bududa_post.groupby("groupnumber", as_index = False)["land_management"].mean()

bududa_lm_delta = pd.DataFrame({"group": bududa_post_lm["groupnumber"].astype(int).astype(str), "delta": bududa_post_lm["land_management"] - bududa_pre_lm["land_management"]})
bududa_lm_delta["id"] = "dp2_group" + bududa_lm_delta["group"] + "_session1"
bududa_lm_delta = bududa_lm_delta.drop(["group"], axis = 1)

# Population pressure
bududa_pre["population"] = bududa_pre[["125_Buildroads_accessmarkets", "126_Government_morebridges", "127_Government_raisenarrowbridges", "128_Newbuildings_highfloors", "129_Communities_ladders", "130_Government_oneclassschools", "131_Commties_girlsandboys", "132_Commties_technicalschools", "133_Government_enforceminimumageof18", "134_Resources_planningsizeoffamilies", "135_Education_familyplanning", "136_HealthcenterIIs", "137_Moreroads_fewerbridges"]].sum(axis = 1) / 13

bududa_post["population"] = bududa_post[["125_Buildroads_accessmarkets", "126_Government_morebridges", "127_Government_raisenarrowbridges", "128_Newbuildings_highfloors", "129_Communities_ladders", "130_Government_oneclassschools", "131_Commties_girlsandboys", "132_Commties_technicalschools", "133_Government_enforceminimumageof18", "134_Resources_planningsizeoffamilies", "135_Education_familyplanning", "136_HealthcenterIIs", "137_Moreroads_fewerbridges"]].sum(axis = 1) / 13

bududa_pre_pp = bududa_pre.groupby("groupnumber", as_index = False)["population"].mean()
bududa_post_pp = bududa_post.groupby("groupnumber", as_index = False)["population"].mean()

bududa_pp_delta = pd.DataFrame({"group": bududa_post_lm["groupnumber"].astype(int).astype(str), "delta": bududa_post_pp["population"] - bududa_pre_pp["population"]})
bududa_pp_delta["id"] = "dp2_group" + bududa_pp_delta["group"] + "_session2"
bududa_pp_delta = bududa_pp_delta.drop(["group"], axis = 1)

# Resettlement
bududa_pre["resettlement"] = bududa_pre[["101_Rezoning", "102_Compesation", "103_Resettle_hostfamilies", "104_support_hostfamilies", "105_Strengthen_DMCs", "106_Raisefunds_DMCs", "107_Training_DMCs", "108_Buildperi-urbancenterrs", "109_Newperi-urbancenters_nearby"]].sum(axis = 1) / 9

bududa_post["resettlement"] = bududa_post[["101_Rezoning", "102_Compesation", "103_Resettle_hostfamilies", "104_support_hostfamilies", "105_Strengthen_DMCs", "106_Raisefunds_DMCs", "107_Training_DMCs", "108_Buildperi-urbancenterrs", "109_Newperi-urbancenters_nearby"]].sum(axis = 1) / 9

bududa_pre_re = bududa_pre.groupby("groupnumber", as_index = False)["resettlement"].mean()
bududa_post_re = bududa_post.groupby("groupnumber", as_index = False)["resettlement"].mean()

bududa_re_delta = pd.DataFrame({"group": bududa_post_lm["groupnumber"].astype(int).astype(str), "delta": bududa_post_re["resettlement"] - bududa_pre_re["resettlement"]})
bududa_re_delta["id"] = "dp2_group" + bududa_re_delta["group"] + "_session3"
bududa_re_delta = bududa_re_delta.drop(["group"], axis = 1)

## Onto the Butaleja

# getting butaleja groups
butaleja_groups = pd.read_csv("data/raw/surveys/butaleja_groups.csv")

# filtering to just the Butaleja data, also joining on groups
butaleja_pre = bududa_and_butaleja[((bududa_and_butaleja["000ID"].str.contains("BUT")) & (bududa_and_butaleja["001_Poll"] == 1))].merge(butaleja_groups, how = "left", left_on = "000ID", right_on = "id")
butaleja_post = bududa_and_butaleja[((bududa_and_butaleja["000ID"].str.contains("BUT")) & (bududa_and_butaleja["001_Poll"] == 2))].merge(butaleja_groups, how = "left", left_on = "000ID", right_on = "id")

# Land Management
butaleja_pre["land_management"] = butaleja_pre[["114_Planttrees_protectriverbeds", "115_Riverchannels_localgovernment", "116_wetlands_dryseason", "117_Riceschemes_notinwetlands", "118_Communities_maintainwaterchannels", "119_Communities_benefitscropdiversity", "120_Communities_de-silting", "121_Government_assistdesilting", "122_Communities_sanitationdrains", "123_Government_drillingcleanwater", "124_Communities_resourcesaccesswater"]].sum(axis = 1) / 11

butaleja_post["land_management"] = butaleja_post[["114_Planttrees_protectriverbeds", "115_Riverchannels_localgovernment", "116_wetlands_dryseason", "117_Riceschemes_notinwetlands", "118_Communities_maintainwaterchannels", "119_Communities_benefitscropdiversity", "120_Communities_de-silting", "121_Government_assistdesilting", "122_Communities_sanitationdrains", "123_Government_drillingcleanwater", "124_Communities_resourcesaccesswater"]].sum(axis = 1) / 11

butaleja_pre_lm = butaleja_pre.groupby("groupnumber", as_index = False)["land_management"].mean()
butaleja_post_lm = butaleja_post.groupby("groupnumber", as_index = False)["land_management"].mean()

butaleja_lm_delta = pd.DataFrame({"group": butaleja_post_lm["groupnumber"].astype(int).astype(str), "delta": butaleja_post_lm["land_management"] - butaleja_pre_lm["land_management"]})
butaleja_lm_delta["id"] = "dp3_group" + butaleja_lm_delta["group"] + "_session1"
butaleja_lm_delta = butaleja_lm_delta.drop(["group"], axis = 1)

# Population pressure
butaleja_pre["population"] = butaleja_pre[["125_Buildroads_accessmarkets", "126_Government_morebridges", "127_Government_raisenarrowbridges", "128_Newbuildings_highfloors", "129_Communities_ladders", "130_Government_oneclassschools", "131_Commties_girlsandboys", "132_Commties_technicalschools", "133_Government_enforceminimumageof18", "134_Resources_planningsizeoffamilies", "135_Education_familyplanning", "136_HealthcenterIIs", "137_Moreroads_fewerbridges"]].sum(axis = 1) / 13

butaleja_post["population"] = butaleja_post[["125_Buildroads_accessmarkets", "126_Government_morebridges", "127_Government_raisenarrowbridges", "128_Newbuildings_highfloors", "129_Communities_ladders", "130_Government_oneclassschools", "131_Commties_girlsandboys", "132_Commties_technicalschools", "133_Government_enforceminimumageof18", "134_Resources_planningsizeoffamilies", "135_Education_familyplanning", "136_HealthcenterIIs", "137_Moreroads_fewerbridges"]].sum(axis = 1) / 13

butaleja_pre_pp = butaleja_pre.groupby("groupnumber", as_index = False)["population"].mean()
butaleja_post_pp = butaleja_post.groupby("groupnumber", as_index = False)["population"].mean()

butaleja_pp_delta = pd.DataFrame({"group": butaleja_post_lm["groupnumber"].astype(int).astype(str), "delta": butaleja_post_pp["population"] - butaleja_pre_pp["population"]})
butaleja_pp_delta["id"] = "dp3_group" + butaleja_pp_delta["group"] + "_session2"
butaleja_pp_delta = butaleja_pp_delta.drop(["group"], axis = 1)

# Resettlement
butaleja_pre["resettlement"] = butaleja_pre[["101_Rezoning", "102_Compesation", "103_Resettle_hostfamilies", "104_support_hostfamilies", "105_Strengthen_DMCs", "106_Raisefunds_DMCs", "107_Training_DMCs", "108_Buildperi-urbancenterrs", "109_Newperi-urbancenters_nearby"]].sum(axis = 1) / 9

butaleja_post["resettlement"] = butaleja_post[["101_Rezoning", "102_Compesation", "103_Resettle_hostfamilies", "104_support_hostfamilies", "105_Strengthen_DMCs", "106_Raisefunds_DMCs", "107_Training_DMCs", "108_Buildperi-urbancenterrs", "109_Newperi-urbancenters_nearby"]].sum(axis = 1) / 9

butaleja_pre_re = butaleja_pre.groupby("groupnumber", as_index = False)["resettlement"].mean()
butaleja_post_re = butaleja_post.groupby("groupnumber", as_index = False)["resettlement"].mean()

butaleja_re_delta = pd.DataFrame({"group": butaleja_post_lm["groupnumber"].astype(int).astype(str), "delta": butaleja_post_re["resettlement"] - butaleja_pre_re["resettlement"]})
butaleja_re_delta["id"] = "dp3_group" + butaleja_re_delta["group"] + "_session3"
butaleja_re_delta = butaleja_re_delta.drop(["group"], axis = 1)

### Merge deltas
all_deltas = ghana_qfood_delta.append(ghana_qwater_delta).append(bududa_lm_delta).append(bududa_pp_delta).append(bududa_re_delta).append(butaleja_lm_delta).append(butaleja_pp_delta).append(butaleja_re_delta)

all_deltas.to_csv("data/processed/survey/opinion_deltas.csv")
