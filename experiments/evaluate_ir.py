from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
import string

from sklearn.metrics import average_precision_score
import numpy as np
import os
import faiss
from tqdm import tqdm
import json
import argparse
from nltk.tokenize import RegexpTokenizer
from collection_index import BM25Index, DenseIndex, reciprocal_rank_fusion, get_sentences_questions, get_contexts_questions_answers

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from nltk.stem import WordNetLemmatizer
import re
import unicodedata
from nltk.tokenize import TweetTokenizer


class IREval:
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    regex_tokenizer = RegexpTokenizer(r'\w+')

    pattern = re.compile(r'(\b\w+)-(\w+\b)')
    quote_pattern = re.compile(r'["`´\/]')
    tknzr = TweetTokenizer()


    def __init__(self, model_name, input_file, top_k):
        self.similarity_model_name = model_name
        self.input_file = input_file
        self.top_k = top_k

    def load_data(self):
        logger.info('loading data')
        with open(self.input_file, 'r') as file:
            json_list = list(file)
        
        data = []
        # [1:] removes the header
        for item in json_list[1:]:
            item = json.loads(item)
            data.append(item)

        logger.info(f"Total samples loaded: {len(data)}")
        return data

 
    def evaluate(self, use_rank_fusion):
        data = self.load_data()
        logger.info('preprocessing data')        

        data = self.load_data()
        sentences, questions, q2s = get_sentences_questions(data)

        logger.info('creating bm25 index')
        bm25index = BM25Index(sentences)

        
        #if use_rank_fusion:
        logger.info('creating dense index')
        denseindex = DenseIndex(sentences, self.similarity_model_name)


        logger.info('Searching for contexts in indexes...')
        
        reciprocal_ranks = []

        for idx, question in enumerate(tqdm(questions, total=len(questions))):
            found = False
            idx_correct = q2s[idx]
            if len(idx_correct) ==0:
                continue


            if use_rank_fusion:
                dense_hits = denseindex.search(question, self.top_k)
            
            text_hits = bm25index.search(question, self.top_k)
                
            
            if use_rank_fusion:
                hits = reciprocal_rank_fusion(dense_hits, text_hits)
            else:
                hits = text_hits

            for rank, hit in enumerate(hits):
                if hit in idx_correct:
                    reciprocal_ranks.append(1/(rank+1))
                    found = True
                    break
            if not found:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Correct Passage Retrieval')
    parser.add_argument('--similarity_model_path', type=str, help='Name or path of the pretrained sentence-bert model to use',
                        default="vabatista/sbert_squad_en_nli")
    parser.add_argument('--input_file', type=str, help='Path to the SQuAD like data file')
    parser.add_argument('--top_k', type=int, default=100, help='top_k sentences to search')
    parser.add_argument('--use_rank_fusion', action='store_true', default=False, help='Use rank fusion to combine results from BM25 and SBERT')

    args = parser.parse_args()

    ir_eval = IREval(args.similarity_model_path, args.input_file, args.top_k)
    mrr = ir_eval.evaluate(args.use_rank_fusion)

    logger.info(f"MRR@{args.top_k} for {args.input_file}: {mrr}")

