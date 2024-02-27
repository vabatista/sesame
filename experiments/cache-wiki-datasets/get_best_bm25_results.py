from sentence_transformers import SentenceTransformer
import random

from tqdm import tqdm
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
from datasets import load_dataset
#enwiki-paragraphs
#ds = load_dataset("castorini/odqa-wiki-corpora", "wiki-text-6-3-tamber", split="train", trust_remote_code=True)

#ssearcher = LuceneSearcher.from_prebuilt_index('wiki-text-6-3-tamber')
ssearcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')

input_file = '/home/users/vabatista/bkp/phd-thesis/qa-datasets/datasets/originals/popqa_longtail.jsonl'

#input_file = '/home/users/vabatista/bkp/phd-thesis/qa-datasets/datasets/originals/TriviaQA-without-context.jsonl'
dataset_name = 'PopQA-longtail-paragraphs'

top_k = 100

def load_data(input_file):
    with open(input_file, 'r') as file:
        json_list = list(file)
    
    data = []
    # [1:] removes the header
    for item in json_list:
        item = json.loads(item)
        data.append(item)
    
    print(f"Total samples loaded: {len(data)}")
    return data

def encode_sentence(model, sentence):
    encoded = model.encode(sentence, show_progress_bar=False)

    encoded = encoded / np.linalg.norm(encoded)
    encoded = np.expand_dims(encoded, axis=0)
    return encoded

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

content_index = {}
best_bm25_results = {}

dataset = load_data(input_file)


## Make a BM25 search for each question and save the top k results
print('Searching for the top k results for each question')
for q_id, item in tqdm(enumerate(dataset), total=len(dataset)):
    question = item['question']
    hits = ssearcher.search(question, top_k)
    best_hits = [{'id': hits[i].docid, 'bm25_score': hits[i].score} for i in range(0, max(top_k, len(hits)))]
    best_bm25_results[q_id] = best_hits
    for hit in hits:
        if hit.docid not in content_index:
            # get the content from ds
            content_index[hit.docid] = hit.lucene_document.get('raw')

print('Saving best bm25 results')
with open(f'{dataset_name}-best-bm25-results.json', 'w') as fp:
    json.dump(best_bm25_results, fp)

with open(f'{dataset_name}-content-index.json', 'w') as fp:
    json.dump(content_index, fp)
    
## Make a Dense reranking using sentence-transformers
print('Re-raanking the top k results using sentence-transformers')
model = SentenceTransformer('vabatista/sbert-mpnet-base-bm25-hard-neg')
for question_id in tqdm(best_bm25_results.keys()):
    for i, doc in enumerate(best_bm25_results[question_id]):
        encoded_sentence = encode_sentence(model, content_index[doc['id']])
        encoded_question = encode_sentence(model, dataset[question_id]['question'])
        best_bm25_results[question_id][i]['dense_score'] = np.dot(encoded_sentence, encoded_question.T)[0][0]
    ## Now sort the results by the dense score
    bm25_hits_list = [hit['id'] for hit in best_bm25_results[question_id]]
    dense_hits_list = [hit['id'] for hit in sorted(best_bm25_results[question_id], key=lambda x: x['dense_score'], reverse=True)]
    best_bm25_results[question_id] = reciprocal_rank_fusion(bm25_hits_list, dense_hits_list)

print('Saving the results')
with open(f'{dataset_name}-reranked-results.json', 'w') as fp:
    json.dump(best_bm25_results, fp)

