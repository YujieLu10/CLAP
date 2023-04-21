import pandas as pd
import json

df = pd.read_csv('../../data/wikihow/wikicausal.csv', delimiter='^')
wikihow_available_examples_list = []
example_idx = 0
for pair_title, pair_headline in zip(df.title.copy(), df.headline.copy()):
    example_idx += 1
    action_txt = ""
    for idx, action in enumerate(str(pair_headline).split("##STEP##")[1:]):
        action_txt += "\nStep {}: ".format(idx+1) + action
    wikihow_available_examples_list.append("Task: " + pair_title + action_txt)
with open("../../data/wikihow/wikihow_chain_of_thought_1000.json", "w") as fp:
    json.dump(wikihow_available_examples_list, fp)