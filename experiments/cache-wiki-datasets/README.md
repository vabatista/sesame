## Caching Wikipedia Dump

This script needs another virtualenv with pyserini, transformers and sentence-transformers installed
It will create 3 files as output:
- best_bm25_results: stores a list of best 100 results for each query from input file using cached pyserini wikipedia dump. It stores only id of paragraph and score.
- content_index: a dictionary with paragraph id as key and paragraph content as value.
- reranked-results: final ranking after apply SESAME RRF method.