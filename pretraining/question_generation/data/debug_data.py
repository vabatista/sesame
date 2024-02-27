import torch
from transformers import T5Tokenizer
import random

path_to_tokenizer = '../t5_qg_tokenizer/'
tokenizer = T5Tokenizer.from_pretrained(path_to_tokenizer, legacy=False)

bin_train_data = torch.load('valid_data_qg_t5_en.pt')

def print_example(data, i):
    input_ids = data[i]['source_ids']
    target_ids = data[i]['target_ids']
    print('\nINPUT\n', tokenizer.decode(input_ids, skip_special_tokens=True))
    print('TARGET\n', tokenizer.decode(target_ids, skip_special_tokens=True))



for i in range(20):
    n = random.randint(0, len(bin_train_data))
    print_example(bin_train_data, n)