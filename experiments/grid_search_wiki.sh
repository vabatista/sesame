#!/bin/bash

# List of lr, epochs, and test_split values for grid search


train_input_files=("qa-datasets/datasets/synth/TriviaQA-wiki-5000_synthetic.jsonl")
input_files=("qa-datasets/datasets/originals/TriviaQA-without-context.jsonl")
ranking_files=("qa-datasets/datasets/wiki-cache/TriviaQA-paragraphs-reranked-results.json")
contents_files=("qa-datasets/datasets/wiki-cache/TriviaQA-paragraphs-content-index.json")

train_input_files=("qa-datasets/datasets/synth/PopQA-wiki-synthetic-5000.jsonl")
#input_files=("qa-datasets/datasets/originals/PopQA-without-context.jsonl")
input_files=("qa-datasets/datasets/originals/popqa_longtail.jsonl")
ranking_files=("qa-datasets/datasets/wiki-cache/PopQA-longtail-paragraphs-reranked-results.json")
contents_files=("qa-datasets/datasets/wiki-cache/PopQA-longtail-paragraphs-content-index.json")

lr_values=("1e-5")
epochs_values=("2")
test_split_values=("0.2")
mask_inputs=(false)
temperatures=("0.1")
top_p=("0.8")
top_k=("200")

N=5  # Set the number of iterations

for ((n=0; n<$N; n++)); do

  # Loop through the parameter combinations
  for lr in "${lr_values[@]}"; do
    for epochs in "${epochs_values[@]}"; do
      for test_split in "${test_split_values[@]}"; do
        for ((i=0; i<${#train_input_files[@]}; i++)); do
          for mi in "${mask_inputs[@]}"; do
            for temperature in "${temperatures[@]}"; do
              for tp in "${top_p[@]}"; do
                for tk in "${top_k[@]}"; do
                  mask_input_param=""
                  if [ "$mi" = true ]; then
                    mask_input_param="--mask_inputs"
                  fi
                  echo "Running experiment with lr=$lr, epochs=$epochs, test_split=$test_split, train_input_file=${train_input_files[i]}, input_file=${input_files[i]} and mask_inputs=$mi and temperature=$temperature and top_p=$tp and top_k=$tk"
                  # saved-models/stabilityai/stablelm-tuned-alpha-7b
                  # saved-models/meta-llama/Llama-2-7b-chat-hf
                  # saved-models/meta-llama/Llama-2-7b-hf
                  #  saved-models/meta-llama/Llama-2-13b-chat-hf
                  # saved-models/mistralai/Mistral-7B-Instruct-v0.2
                  python experiments/full_experiment.py \
                    --base_model_name saved-models/meta-llama/Llama-2-13b-chat-hf \
                    --train_input_file ${train_input_files[i]} \
                    --epochs $epochs \
                    --similarity_model_path vabatista/sbert-mpnet-base-bm25-hard-neg \
                    --input_file ${input_files[i]} \
                    --top_k_contexts 5 \
                    --test_split $test_split \
                    --lr $lr \
                    --temperature $temperature \
                    --top_p $tp \
                    --top_k $tk \
                    --wiki --llm --do_train --do_eval --do_search --quantize ${mask_input_param} \
                    --ranking_file ${ranking_files[i]} \
                    --contents_file ${contents_files[i]} 
                  # echo "==============> Experiment with lr=$lr, epochs=$epochs, test_split=$test_split, train_input_file=${train_input_files[i]}, input_file=${input_files[i]} completed"
                done
              done
            done
          done
        done
      done
    done
  done
done