#!/bin/bash

# List of lr, epochs, and test_split values for grid search

# "qa-datasets/synth/squad2/t5-base/DuoRC.ParaphraseRC-dev_synthetic.json" "qa-datasets/synth/squad2/t5-base/TriviaQA-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/NaturalQuestionsShort-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/TextbookQA-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/DROP-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/HotpotQA-dev_synthetic.json"

#train_input_files=("qa-datasets/synth/squad2/t5-large/NewsQA-dev_synthetic.json" "qa-datasets/synth/squad2/t5-base/TriviaQA-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/NaturalQuestionsShort-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/TextbookQA-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/HotpotQA-dev_synthetic.json")
#input_files=("qa-datasets/originals/dev/NewsQA-dev.jsonl" "qa-datasets/originals/dev/TriviaQA-web-dev.jsonl" "qa-datasets/originals/dev/NaturalQuestionsShort.jsonl" "qa-datasets/originals/TextbookQA-dev_modified.jsonl" "qa-datasets/originals/dev/HotpotQA-dev.jsonl"  )

#train_input_files=("qa-datasets/synth/squad2/t5-large/DROP-dev.json" "qa-datasets/ae-qg-avgl-synth/BioASQ-dev_synthetic.json")
#input_files=("qa-datasets/originals/DROP-dev_modified.jsonl" "qa-datasets/originals/BioASQ-dev.jsonl")

#train_input_files=("qa-datasets/ae-qg-avgl-synth/DROP-dev_synthetic.json" )
#input_files=("qa-datasets/originals/DROP-dev_modified.jsonl")

#train_input_files=("qa-datasets/ae-qg-avgl-synth/HotpotQA-dev_synthetic.json" )
#input_files=("qa-datasets/originals/dev/HotpotQA-dev.jsonl")

#train_input_files=("qa-datasets/synth/squad2/t5-large/NewsQA_synthetic.json")
#input_files=("qa-datasets/originals/dev/NewsQA-dev.jsonl")

#train_input_files=("qa-datasets/synth/squad2/t5-base/TriviaQA-dev_synthetic.json")
#input_files=("qa-datasets/originals/dev/TriviaQA-web-dev.jsonl")

#train_input_files=("qa-datasets/ae-qg-avgl-synth/NaturalQuestionsShort-dev_synthetic.json" )
#input_files=("qa-datasets/originals/dev/NaturalQuestionsShort.jsonl")

#train_input_files=("qa-datasets/ae-qg-avgl-synth/TextbookQA-dev_synthetic.json")
#input_files=("qa-datasets/originals/TextbookQA-dev_merged.jsonl")

#train_input_files=("qa-datasets/datasets/synth/faquad_synthetic.json")
#input_files=("qa-datasets/datasets/originals/faquad_modified.jsonl")

train_input_files=("qa-datasets/ae-qg-avgl-synth/BioASQ-dev_synthetic.json")
input_files=("qa-datasets/originals/BioASQ-dev.jsonl")

#train_input_files=("qa-datasets/datasets/synth/rgb_benchmark_synthetic.jsonl" )
#input_files=("qa-datasets/datasets/originals/rgb_benchmark_60-noise.jsonl")

## all english
#train_input_files=("qa-datasets/ae-qg-avgl-synth/DROP-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/BioASQ-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/TextbookQA-dev_synthetic.json" "qa-datasets/ae-qg-avgl-synth/NaturalQuestionsShort-dev_synthetic.json" "qa-datasets/synth/squad2/t5-base/TriviaQA-dev_synthetic.json" "qa-datasets/synth/squad2/t5-large/NewsQA_synthetic.json" "qa-datasets/ae-qg-avgl-synth/HotpotQA-dev_synthetic.json")
#input_files=("qa-datasets/originals/DROP-dev_modified.jsonl" "qa-datasets/originals/BioASQ-dev.jsonl" "qa-datasets/originals/TextbookQA-dev_merged.jsonl" "qa-datasets/originals/dev/NaturalQuestionsShort.jsonl" "qa-datasets/originals/dev/TriviaQA-web-dev.jsonl" "qa-datasets/originals/dev/NewsQA-dev.jsonl" "qa-datasets/originals/dev/HotpotQA-dev.jsonl" )



lr_values=("1e-3")
epochs_values=("1")
test_split_values=("0.4" "0.6" "0.8") 
mask_inputs=(false)
temperatures=("0.1")
top_p=("0.8")
top_k=("200")
#models=("saved-models/meta-llama/Llama-2-7b-chat-hf" "saved-models/meta-llama/Llama-2-13b-chat-hf")
N=5 # Set the number of iterations

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
                  # saved-models/falcon-7b-instruct
                  # maritaca-ai/sabia-7b
                  python experiments/full_experiment.py \
                    --base_model_name saved-models/meta-llama/Llama-2-7b-chat-hf \
                    --train_input_file ${train_input_files[i]} \
                    --epochs ${epochs} \
                    --similarity_model_path vabatista/sbert-mpnet-base-bm25-hard-neg \
                    --input_file ${input_files[i]} \
                    --top_k_contexts 5 \
                    --test_split $test_split \
                    --lr $lr  \
                    --temperature $temperature \
                    --top_p $tp \
                    --top_k $tk \
                    --language en \
                    --llm --do_search --do_train --do_eval ${mask_input_param}
                    #--quantize
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