import pandas as pd
from icecream import ic
import argparse
import json
from random import randrange

parser = argparse.ArgumentParser(description='Intervention')
parser.add_argument('--data_type', choices=['wikihow', 'robothow'], default='robothow', help='choices')
parser.add_argument('--intervene_type', choices=['conf', 'how' ,'goal'], default='conf', help='choices')
args = parser.parse_args()

intervene_configuration_list = [
    "livingroom",
    "bathroom",
    "bedroom",
    "dining room",
    "home office",
    "kitchen"
]

with open('../../data/{}/{}_available_examples.json'.format(args.data_type, args.data_type), 'r') as f:
    available_examples = json.load(f)

example_data_list = []
for example in available_examples[:60]:
    task = example.split('\n')[0]
    if args.intervene_type == "conf":
        random_location = intervene_configuration_list[randrange(len(intervene_configuration_list))]
        intervened_program_str = task + f' in {random_location}\n'
        new_step_str = "Step 1: Walk to " + random_location + '\n'
        origin_step = example.split('\n')[1]
        isLocationStep = any(item in origin_step for item in intervene_configuration_list)
        if isLocationStep:
            # replace original first step, if it's related to navigate to a certain location
            intervened_program_str = intervened_program_str + new_step_str + '\n'.join(example.split('\n')[2:])
        else:
            # directly prepend the navigation step before the original steps
            intervened_program_str = intervened_program_str + new_step_str + '\n'.join(example.split('\n')[1:])
    elif args.intervene_type == "how":
        import random
        how_step_str = random.choice(example.split('\n')[1:])
        intervened_program_str = task + ' ({})'.format(how_step_str[how_step_str.index(':')+2:]) + '\n' + '\n'.join(example.split('\n')[1:])
    elif args.intervene_type == "goal": #composite task
        import random
        composite_example = random.choice(available_examples)
        intervened_program_str = task + ' and ' +  composite_example.split('\n')[0] + '\n' + '\n'.join(example.split('\n')[1:]) + '\n' + '\n'.join(composite_example.split('\n')[1:])
    example_data_list.append(intervened_program_str)

with open('../../data/{}/{}_available_examples_inter{}.json'.format(args.data_type, args.data_type, args.intervene_type), 'w') as resultfile:
    import simplejson
    simplejson.dump(example_data_list, resultfile)
