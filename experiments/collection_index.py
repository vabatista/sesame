import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import string
import faiss
import re
import unicodedata
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize
import json
import os
from tqdm import tqdm 

def reciprocal_rank_fusion(list1, list2):
    # Combine the two lists
    combined_list = list(set(list1 + list2))

    # Initialize a dictionary to store reciprocal ranks
    rr_dict = {}

    # Calculate reciprocal ranks for list1
    for i, doc_id in enumerate(list1, start=1):
        rr_dict[doc_id] = rr_dict.get(doc_id, 0) + (1 / i)

    # Calculate reciprocal ranks for list2
    for i, doc_id in enumerate(list2, start=1):
        rr_dict[doc_id] = rr_dict.get(doc_id, 0) + 1 / i

    # Sort the combined list based on the sum of reciprocal ranks
    fused_list = sorted(combined_list, key=lambda x: rr_dict.get(x, 0), reverse=True)

    return fused_list


def get_sentences_questions(squad_data):
    sentences = []
    questions = []
    q2s = {}
    last_sent_idx = 0
    cur_q_idx = -1
    for data in squad_data:
        last_sent_idx = len(sentences)
        if 'context' in data:
            context = data['context']
        else: 
            context = ' '
        context_sentences = sent_tokenize(context, language='english')        
        sentences.extend(context_sentences)
        if 'qas' in data:
            for qa in data['qas']:
                question = qa['question']
                #answer_starts = [start[0] for start in qa['detected_answers'][0]['char_spans']]
                answer_starts = []
                for ans in qa['detected_answers']:
                    if 'char_spans' in ans:
                        answer_starts.append(ans['char_spans'][0][0])
                    elif 'answer_start' in ans:
                        answer_starts.append(ans['answer_start'])
                cur_q_idx += 1
                questions.append(question)
                q2s[cur_q_idx] = []
                
                # Find the sentence containing the answer span
                for idx, s in enumerate(context_sentences):
                    start_index = context.find(s)
                    end_index = start_index + len(s)
                    for answer_start in answer_starts:
                        if start_index <= answer_start < end_index:
                            q2s[cur_q_idx].append(idx + last_sent_idx)
        else:
            questions.append(data['question'])
            q2s[cur_q_idx] = []
            cur_q_idx += 1

    return sentences, questions, q2s

def get_contexts_questions_answers(json_data):
    data_list = []

    for paragraph in json_data:
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

class Index:
    '''
    This class is used to create an index for the collection of documents.
    '''

    def load_data(self, input_file):
        '''
        Assumes that the format of the input file is a jsonl from TREC 2019
        '''
        with open(input_file, 'r') as file:
            json_list = list(file)
        
        data = []
        # [1:] removes the header
        for item in json_list[1:]:
            item = json.loads(item)
            data.append(item)

        return data

 

class BM25Index(Index):

    pattern = re.compile(r'(\b\w+)-(\w+\b)')
    quote_pattern = re.compile(r'["`´\/]')
    tknzr = TweetTokenizer()

    def __init__(self, sentences, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            print('Loading BM25 index from cache')
            with open(cache_path, 'rb') as f:
                self.bm25 = pickle.load(f)
        else:
            self.bm25 = self.create_bm25_index(sentences)
            if cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.bm25, f)


    def strip_accents(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')

    def get_words(self, sentence):
        sentence = self.strip_accents(sentence)
        sentence = re.sub(r'[^\x00-\x7F]+',' ', sentence)

        #remove quotes
        sentence = self.quote_pattern.sub(' ', sentence)
        
        #remove hyphens
        sentence = self.pattern.sub(r'\1 \2', sentence)

        #words = word_tokenize(sentence)
        words = self.tknzr.tokenize(sentence)

        filtered_sentence = [w.lower() for w in words if w.lower() not in string.punctuation] 
        filtered_sentence = [w[:-1] if w.endswith('.') else w for w in filtered_sentence ]
        return filtered_sentence

    def create_bm25_index(self, sentences):
        tokenized_sentences = [self.get_words(sentence) for sentence in tqdm(sentences)]
        bm25 = BM25Okapi(tokenized_sentences, k1=0.1, b=0.1)
        return bm25

    def search(self, query, k):
        tokenized_query = self.get_words(query)
        scores = self.bm25.get_scores(tokenized_query)
        enumerated_list = list(enumerate(scores))

        # Sort the list based on the values in descending order
        sorted_list = sorted(enumerated_list, key=lambda x: x[1], reverse=True)

        # Extract the indices of the top k elements
        top_k_indices = [index for index, _ in sorted_list[:k]]
        return list(top_k_indices)


class DenseIndex(Index):
    def __init__(self, sentences, similarity_model, cache_path=None):
        self.similarity_model = SentenceTransformer(similarity_model)


        if cache_path and os.path.exists(cache_path):
            print('Loading Dense index from cache')
            self.index = faiss.read_index(cache_path)
        else:
            self.index = self.create_index(sentences)
            if cache_path:
                faiss.write_index(self.index, cache_path)

    def create_index(self, sentences):
        corpus_embeddings = self.similarity_model.encode(sentences, show_progress_bar=True, convert_to_numpy=True, batch_size=512)
        embedding_size = 768  # Size of embeddings ## Fixed for the model type used here

        # Defining our FAISS index
        n_clusters = int(len(sentences) ** 0.5)
        quantizer = faiss.IndexFlatIP(embedding_size)
        index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 50

        # Normalize vectors to unit length
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]

        # Train the index to find a suitable clustering
        index.train(corpus_embeddings)

        # Add all embeddings to the index
        index.add(corpus_embeddings)

        return index

    def search(self, query, k):
        question_embedding = self.similarity_model.encode(query, show_progress_bar=False)
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)
        distances, corpus_ids = self.index.search(question_embedding, k=k)

        dense_hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        dense_hits = sorted(dense_hits, key=lambda x: x['score'], reverse=True)
        dense_hits = [hit['corpus_id'] for hit in dense_hits]

        return dense_hits