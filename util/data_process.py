from unittest import skip
from matplotlib.style import available
import pandas as pd
from icecream import ic

# ======== WikiHow Available Actions Construction
df = pd.read_csv('../../data/wikihow/wikicausal_2000.csv')
print(df.keys())
titles = df.title.copy()
headlines = df.headline.copy()

wikihow_available_actions_set = set()

action_count = 0
for headline in headlines:
    for action in str(headline).split("##STEP##")[1:]:
        clean_action = str(action).strip().lower()[:-2]
        if len(clean_action) <= 0: continue
        action_count += 1
        wikihow_available_actions_set.add(clean_action)


ic(len(headlines), action_count, len(wikihow_available_actions_set))

import json
with open("../../data/wikihow/wikihow_available_actions_2000.json", "w") as fp:
    json.dump(list(wikihow_available_actions_set), fp)

# ======== WikiHow Available Examples Construction
import pandas as pd
import json

df = pd.read_csv('../../data/wikihow/wikicausal_2000.csv')
wikihow_available_examples_list = []
for pair_title, pair_headline in zip(df.title.copy(), df.headline.copy()):
    action_txt = ""
    for idx, action in enumerate(str(pair_headline).split("##STEP##")[1:]):
        action_txt += "\nStep {}: ".format(idx+1) + action
    wikihow_available_examples_list.append("Task: " + pair_title + action_txt)
with open("../../data/wikihow/wikihow_available_examples_2000.json", "w") as fp:
    json.dump(wikihow_available_examples_list, fp)
