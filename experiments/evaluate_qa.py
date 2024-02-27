import argparse
import re
import json
import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from collection_index import BM25Index, DenseIndex, reciprocal_rank_fusion, get_sentences_questions
from setup import SesameConfig
from datasets import load_metric
import torch

import string
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QAEvaluator:
    window = 2

    def __init__(self, sesame_config: SesameConfig):
        self.sesame_config = sesame_config

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info('loading finetuned model and tokenizer...')
        model = AutoModelForQuestionAnswering.from_pretrained(self.sesame_config.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.sesame_config.output_dir)
        self.nlp = pipeline("question-answering", model=model, 
                            tokenizer=tokenizer, device=self.device)


    def get_sentence_window(self, sentences, hit, window=1):
        context_before = ' '.join(sentences[max(0, hit - window):hit]) if window > 0 and hit - window >= 0 else ""
        current_sentence = sentences[hit]
        context_after = ' '.join(sentences[hit + 1:hit + window + 1]) if window > 0 and hit + window < len(sentences) else ""

        return context_before + ' ' + current_sentence + ' ' + context_after

        
    
    def normalize_answer(self, s):
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
    def calc_precision(self, predictions, references):
        ##{"id": "0", "answers": {"answer_start": [0], "text": ["The 49ers"]}}
        precision = 0
        for pred, ref in zip(predictions, references):
            is_correct = False
            pred = self.normalize_answer(pred['prediction_text'])
            for answer in ref['answers']['text']:
                n_answer = self.normalize_answer(answer)
                if pred in n_answer or n_answer in pred:
                    is_correct = True
                    break
            if is_correct:
                precision += 1
        return (precision / len(predictions))*100

    def load_data(self):
        logger.info('loading data')
        with open(self.sesame_config.inference_file, 'r') as file:
            json_list = list(file)
        
        data = []
        # [1:] removes the header
        for item in json_list[1:]:
            item = json.loads(item)
            data.append(item)

        logger.info(f"Total samples loaded: {len(data)}")
        return data

    def get_contexts_questions_answers(self, json_data):
        data_list = []

        for paragraph in tqdm(json_data):
            #context = self.clean_context(paragraph['context'])
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
    
    def evaluate(self):


        references = []
        correct_contexts = []
        found_contexts = []
        predictions = []
        predictions_with_correct_context = []

        data = self.load_data()
        logger.info('preprocessing data')        
        sentences, questions, q2s = get_sentences_questions(data)
        cache_path_bm25 = None
        cache_path_faiss = None

        logger.info('creating bm25 index')
        bm25index = BM25Index(sentences, cache_path_bm25)

        logger.info('creating dense index')
        denseindex = DenseIndex(sentences, self.sesame_config.similarity_model_path, cache_path_faiss)

        triples = self.get_contexts_questions_answers(data)

        for idx, triple in enumerate(triples):
            question, answers, context = triple['question'], triple['answer'], triple['context']
            correct_contexts.append(context)
            references.append({'id': str(idx), 'answers': {'answer_start': [context.find(answer) for answer in answers], 
                                                           'text': [answer for answer in answers]}})
        n_samples = len(triples)
        if self.sesame_config.do_search:
            logger.info('Searching for best contexts...')
            for idx, triple in enumerate(tqdm(triples[:n_samples])):
                if bm25index:
                    text_hits = bm25index.search(triple['question'], self.sesame_config.top_k_contexts)
                    dense_hits = denseindex.search(triple['question'], self.sesame_config.top_k_contexts)
                    hits = reciprocal_rank_fusion(dense_hits, text_hits)
                else:
                    hits = denseindex.search(triple['question'], self.sesame_config.top_k_contexts)

                top_k_sentences = ''
                for hit in hits[0:self.sesame_config.top_k_contexts]:
                    top_k_sentences += self.get_sentence_window(sentences, hit, window=self.window)
                found_contexts.append(top_k_sentences)

        logger.info('Generating predictions for correct contexts...')
        result_with_correct_context = self.nlp(question=questions[:n_samples], context=correct_contexts[:n_samples])
        predictions_with_correct_context = [{'id':str(idx), 'prediction_text': result['answer']} 
                                            for idx, result in enumerate(result_with_correct_context)]     
        
        if self.sesame_config.do_search:

            logger.info('Generating predictions for best contexts...')
            result = self.nlp(question=questions[:n_samples], context=found_contexts[:n_samples])
            predictions = [{'id':str(idx), 'prediction_text': pred['answer']} for idx, pred in enumerate(result)]    
            

        return predictions, predictions_with_correct_context, references[:n_samples]


'''
This class is used to evaluate a fine-tuned BERT model on a dataset based on Wikipedia.
I tested only with PopQA and TriviaQA. Original datasets provided a context for each question based on
BM25 results/snipets. 
I used a different ranking model to retrieve top 100 passages from castorini/odqa-wiki-corpora on huggingface
It was necessary to search offline and store results into cache because the index is too large.
'''
class WikiEvaluator(QAEvaluator):
    
    content_dict = {}
    ranks = {}

    def __init__(self, sesame_config: SesameConfig):
        self.sesame_config = sesame_config
       

    
    '''
    For PopQA and TriviaQA only pairs of question and answers are provided.
    '''
    def load_data(self):

        logger.info('loading data')
        with open(self.sesame_config.inference_file, 'r') as file:
            json_list = list(file)
        
        data = []
        # [1:] removes the header, but it is not necessary for PopQA and TriviaQA
        # But I made a mistake creating the cache from the 2nd line, so I need to remove the first to match indexes
        for item in json_list: #[1:]:
            item = json.loads(item)
            data.append(item)

        logger.info(f"Total samples loaded: {len(data)}")
        return data

    def load_contents(self):
        logger.info('loading contents index')
        with open(self.sesame_config.contents_file, 'r') as file:
            contents = json.load(file)
        return contents
    
    def load_ranks(self):
        logger.info('loading ranks index')
        with open(self.sesame_config.ranking_file, 'r') as file:
            ranks = json.load(file)
        return ranks

    def evaluate(self):
        
        references = []
        found_contexts = []
        predictions = []
        questions = []

        data = self.load_data()
        self.content_dict = self.load_contents()
        self.ranks = self.load_ranks()

        not_found_count = 0
        for idx, qa in tqdm(enumerate(data), total=len(data)):
            question, answers = qa['question'], qa['answer']
            if type(answers) != list:
                answers = eval(answers)
            if str(idx) not in self.ranks:
                not_found_count += 1
                continue
            questions.append(question)
            found = []
            for i in range(self.sesame_config.top_k_contexts):
                if type(self.ranks[str(idx)][0]) == str:
                    found.append(self.content_dict[self.ranks[str(idx)][i]])
                else:
                    found.append(self.content_dict[self.ranks[str(idx)][i]['id']])
            found_contexts.append(' '.join(found))

            
            references.append({'id': str(idx), 'answers': {'answer_start': [0]*len(answers), 
                                                           'text': [answer for answer in answers]}})

        logger.info('Generating predictions for best contexts...')
        result = self.nlp(question=questions, context=found_contexts)
        predictions = [{'id':str(idx), 'prediction_text': pred['answer']} for idx, pred in tqdm(enumerate(result), total=len(result))]    
        
        logger.debug(f'Not found ranks for {not_found_count} questions')

        return predictions, [], references
