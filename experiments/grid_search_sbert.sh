#!/bin/bash

# List of lr, epochs, and test_split values for grid search
lr_values=("5e-5")
epochs_values=("1")
#epochs_values=("6" "7" "8" "9" "10")
test_split_values=("0.99")

#train_input_files=("qa-datasets/ae-qg-avgl-synth/DROP-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/BioASQ-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/TextbookQA-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/NaturalQuestionsShort-dev_synthetic.json" "qa-datasets/synth/squad2/t5-base/TriviaQA-dev_synthetic.json" "qa-datasets/synth/squad2/t5-large/NewsQA_synthetic.json" "qa-datasets/ae-qg-avgl-synth/HotpotQA-dev_synthetic.json")
#input_files=("qa-datasets/originals/DROP-dev_modified.jsonl" "qa-datasets/originals/BioASQ-dev.jsonl" "qa-datasets/originals/TextbookQA-dev_merged.jsonl" "qa-datasets/originals/dev/NaturalQuestionsShort.jsonl" "qa-datasets/originals/dev/TriviaQA-web-dev.jsonl" "qa-datasets/originals/dev/NewsQA-dev.jsonl" "qa-datasets/originals/dev/HotpotQA-dev.jsonl" )

train_input_files=("qa-datasets/datasets/synth/faquad_synthetic.json")
input_files=("qa-datasets/datasets/originals/faquad_modified.jsonl")

N=1  # Set the number of iterations

for ((n=0; n<$N; n++)); do
  # Loop through the parameter combinations
  for lr in "${lr_values[@]}"; do
    for epochs in "${epochs_values[@]}"; do
      for test_split in "${test_split_values[@]}"; do
        for ((i=0; i<${#train_input_files[@]}; i++)); do
            echo "Running experiment with lr=$lr, epochs=$epochs, test_split=$test_split, train_input_file=${train_input_files[i]}, input_file=${input_files[i]}"
            # pierreguillou/bert-base-cased-squad-v1.1-portuguese
            # mrm8488/spanbert-finetuned-squadv2
            # timpal0l/mdeberta-v3-base-squad2
            python experiments/full_experiment.py \
              --base_model_name pierreguillou/bert-base-cased-squad-v1.1-portuguese \
              --train_input_file ${train_input_files[i]} \
              --epochs $epochs \
              --similarity_model_path vabatista/sbert-mpnet-base-bm25-hard-neg-pt-br \
              --input_file ${input_files[i]} \
              --top_k_contexts 5 \
              --test_split $test_split \
              --lr $lr --do_train --do_eval
            # echo "==============> Experiment with lr=$lr, epochs=$epochs, test_split=$test_split, train_input_file=${train_input_files[i]}, input_file=${input_files[i]} completed"
          done
        done
      done
    done
  done
done