#!/bin/bash

input_files=("qa-datasets/originals/DROP-dev_modified.jsonl" "qa-datasets/originals/BioASQ-dev.jsonl" "qa-datasets/originals/dev/HotpotQA-dev.jsonl" "qa-datasets/originals/dev/NewsQA-dev.jsonl" "qa-datasets/originals/dev/TriviaQA-web-dev.jsonl" "qa-datasets/originals/TextbookQA-dev_modified.jsonl" "qa-datasets/originals/dev/NaturalQuestionsShort.jsonl" "qa-datasets/originals/dev/NaturalQuestionsShort.jsonl")

# Loop through the parameter combinations
for ((i=0; i<${#input_files[@]}; i++)); do
  echo "Running experiment with input_file=${input_files[i]}"
  # pierreguillou/bert-base-cased-squad-v1.1-portuguese
  # mrm8488/spanbert-finetuned-squadv2
  # timpal0l/mdeberta-v3-base-squad2
  
  python experiments/evaluate_qa.py \
    --qa_model_path mrm8488/spanbert-finetuned-squadv2 \
    --similarity_model_path vabatista/sbert-mpnet-base-bm25-hard-neg \
    --input_file ${input_files[i]} \
    --top_k 5 \
    --base_model_name mrm8488/spanbert-finetuned-squadv2 \
    --do_search \
    --results_path results/baselines 
done
