import pandas as pd
from icecream import ic
import argparse
import json

parser = argparse.ArgumentParser(description='task split')
parser.add_argument('--data_type', choices=['wikihow', 'robothow'], default='robothow', help='choices')
args = parser.parse_args()

available_examples_filepath = '../../data/{}/{}_available_examples.json'.format(args.data_type, args.data_type)
with open(available_examples_filepath, 'r') as f:
    available_examples = json.load(f)

task_set = set()
unique_task_program = []
with open("../../data/{}/{}_available_examples_tasksplit.json".format(args.data_type, args.data_type), "w") as fp:
    for example in available_examples:
        task_name = example.split('\n')[0]
        if task_name in task_set:
            continue
        task_set.add(task_name)
        unique_task_program.append(example)
    json.dump(unique_task_program, fp)
    ic(task_set)