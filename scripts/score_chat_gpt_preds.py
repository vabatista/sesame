## This script just scores ChatGPT predictions

from datasets import load_metric
import json
from nltk import sent_tokenize
from tqdm import tqdm
import re, string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|o)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + '–')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

'''
This function calculates the precision of the predictions. The prediction is
considered accurate if it matches any of the answers in the reference or vice-versa.
'''
def calc_precision(predictions, references):
    ##{"id": "0", "answers": {"answer_start": [0], "text": ["The 49ers"]}}
    precision = 0
    for pred, ref in zip(predictions, references):
        is_correct = False
        if pred['prediction_text'] == None:
            pred = 'NOT ANSWERD'
        else:
            pred = pred['prediction_text']
        pred = normalize_answer(pred)
        for answer in ref['answer']:
            n_answer = normalize_answer(answer)
            if pred in n_answer or n_answer in pred:
                is_correct = True
                break
        if is_correct:
            precision += 1
    return (precision / len(predictions))*100

def load_data(input_file):
    with open(input_file, 'r') as file:
        json_list = list(file)

    data = []
    # [1:] removes the header
    for item in json_list[1:]:
        item = json.loads(item)
        data.append(item)

    return data

def get_contexts_from_squad(json_data):
    contexts = []

    for paragraph in tqdm(json_data):
        for sentence in sent_tokenize(paragraph['context'], language='english'):
            contexts.append(sentence)
    return contexts

def get_contexts_questions_answers(json_data):
    data_list = []

    for paragraph in tqdm(json_data):
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            if qa['detected_answers']:
                answer = list(set([a['text'] for a in qa['detected_answers']]))
            elif qa['answers']:
                answer = list(qa['answers'])

            data_dict = {
                'context': context,
                'question': question,
                'answer': answer
            }
            data_list.append(data_dict)

    return data_list

predictions_file = 'experiments/chatgpt/faquad-all-chatgpt-predictions-correct-context.json'
references_file = 'qa-datasets/datasets/originals/faquad_modified.jsonl'


data = load_data(references_file)
references = get_contexts_questions_answers(data)


with open(predictions_file, 'r') as file:
    predictions = json.load(file)

if len(predictions) == len(references):
    print(calc_precision(predictions=predictions, references=references))
