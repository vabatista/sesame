## This script generates the RGB benchmark dataset with X% of noise from the original file from authors.

import random
import math
import numpy as np 
import json
from tqdm import tqdm

with open('/rgb-benchmark/en.json', 'r') as f:
    json_list = list(f)

noise_rate = 0.6

rgb_dataset = [{"header": {"dataset": "RGB-en", "split": "full"}}]
for json_str in json_list:
    item = json.loads(json_str)
    
    neg_num = math.ceil(len(item['positive']) * noise_rate)
    pos_num = len(item['positive']) - neg_num

    triple = {}
    ans_list = item['answer']
    if type(ans_list) == str:
        ans_list = [ans_list]
    elif type(ans_list[0]) == list:
        ans_list = ans_list[0]
    triple["qas"] = [{"question": item['query'], "detected_answers": [{'text': txt} for txt in ans_list]}]
    positive = random.sample(item['positive'], min(pos_num, len(item['positive'])))
    negative = random.sample(item['negative'], min(neg_num, len(item['negative'])))
    all_ex = positive + negative
    random.shuffle(all_ex)
    
    triple["context"] = " ".join(all_ex)
    rgb_dataset.append(triple)


with open('rgb_benchmark_60-noise.jsonl', 'w') as outfile:
    for entry in rgb_dataset:
        json.dump(entry, outfile)
        outfile.write('\n')