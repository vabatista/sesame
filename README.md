# <img src="sesame_logo.png" width="150"/> Self-supervised framework for Extractive queStion Answering over docuMent collEctions


## Installation instructions

1 - Create a mamba/conda environment with the following command:

```mamba create -n sesame python=3.11```

2 - Activate the environment with the following command:

```conda activate sesame```

3 - Install dependencies with the following commands:

```
pip install -r requirements.txt
conda install -c conda-forge faiss-gpu # for some reason it does not work with pip
```

## Pretraining foundation models

You don't have to pretrain the foundation models to run experiments. We provide them in HuggingFace model hub. 

* sentence-similrity English [vabatista/sbert-mpnet-base-bm25-hard-neg](https://huggingface.co/vabatista/sbert-mpnet-base-bm25-hard-neg)
* sentence-similarity Portuguese [vabatista/sbert-mpnet-base-bm25-hard-neg-pt-br](https://huggingface.co/vabatista/sbert-mpnet-base-bm25-hard-neg-pt-br)
* Answer Extraction English [vabatista/t5-small-answer-extraction-en](https://huggingface.co/vabatista/t5-small-answer-extraction-en)
* Question Generation English [vabatista/t5-small-question-generation-en](https://huggingface.co/vabatista/t5-small-question-generation-en)
* Answer Extraction Portuguese [vabatista/question-generation-t5-small-pt-br-2](https://huggingface.co/vabatista/question-generation-t5-small-pt-br-2)
* Question Generation Portuguese [vabatista/question-generation-t5-small-pt-br-2](https://huggingface.co/vabatista/question-generation-t5-small-pt-br-2) ** We train the model jointly for QA and AE for portuguese, so same model for both tasks.


However, if you want to pretrain them, you can use the following commands:

### Train SentenceTransformers from scratch

```
 python pretraining/sentence_embedding/finetune-sbert-bi-encoder.py \
    --use_pre_trained_model \
    --epochs 2 \
    --train_file qa-datasets/sbert/squad-2.0-hard-neg-v2.json \
    --model_name sentence-transformers/all-mpnet-base-v2
```


### Train Question-generation models from scratch

Prepare data for training (replace `ans_ext` with `qg` for question generation):

```
python prepare_data.py \
    --task ans_ext \
    --model_type t5 \
    --dataset_path data/squad_v2 \
    --qg_format highlight_qg_format \
    --max_source_length 512 \
    --max_target_length 32 \
    --train_file_name train_data_ans_ext_t5.pt \
    --valid_file_name valid_data_ans_ext_t5.pt \
    --tokenizer tokenizer_path_or_id \
    --valid_for_qg_only 
```    

Train T5 models (replace `ans_ext` with `qg` for question generation):

```
python run_qg.py \
    --model_name_or_path t5-small \
    --model_type t5 \
    --tokenizer_name_or_path t5_qg_tokenizer \
    --output_dir ./saved-models/t5-small-ans-ext \
    --train_file_path data/train_data_ans_ext_t5.pt \
    --valid_file_path data/valid_data_ans_ext_t5.pt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --seed 5 \
    --do_train \
    --do_eval \
    --logging_steps 100 \
    --remove_unused_columns False \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --warmup_ration 0.1
```

## Domain Adaptation

To generate synthetic data from a collection of contexts from a jsonl file, run the following command:

```
python domain_adaptation/generate_synthetic_data.py \
    --t5_model_name vabatista/t5-small-question-generation-en \
    --t5_ans_model_name vabatista/t5-small-answer-extraction-en \
	--input_file ./qa-datasets/SQuAD_modified.jsonl \
	--output_file ./qa-datasets/SQuAD_synthetic.jsonl 
```

You need to adapt this code if your input is in different format from [MRQA 2019 datasets](https://github.com/mrqa/MRQA-Shared-Task-2019).


## Reproduce Experimental Setup

### Run Information Retrivaval Experiment

```
python experiments/evaluate_ir.py \
    --similarity_model_path vabatista/sbert-mpnet-base-bm25-hard-neg \
    --input_file qa-datasets/originals/DROP-dev_modified.jsonl \
    --top_k 10 \
    --use_rank_fusion

```

### Run one full experiment

1. Download the LLama model following these [instructions](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_llama_2.md)

2. Run the following command:

```
python experiments/full_experiment.py \
	--base_model_name saved-models/meta-llama/Llama-2-7b-hf \
	--train_input_file qa-datasets/datasets/synth/DROP-dev.json \
	--epochs 1 \
	--similarity_model_path vabatista/sbert-mpnet-base-bm25-hard-neg \
	--input_file qa-datasets/datasets/originals/DROP-dev_modified.jsonl \
    --top_k_contexts 5 \
    --test_split 0.2 \
    --lr 1e-3  \
    --language en \ # or pt for portuguese
    --do_search --do_train --do_eval \ # the remaining of parameters are for LLaMA
    --temperature 0.1 \
    --top_p 0.8 \
    --top_k 200 \
    --llm \
    --use_hf_ft # This is optional for LLM and uses huggingface fine-tuning instead of lit-gpt. 
```
Huggingface finetuning is slower, but allows you to run with any model from huggingface model hub. LitGPT is faster, uses less memory, but it supports only [some models](https://github.com/Lightning-AI/lit-gpt?tab=readme-ov-file#-lit-gpt-1).
We also created several shell scripts to run a series of experiments in /experiments folder. You can use them as a reference to run your own experiments.

### Datasets Download

Download the orignal, our synthetic, wikipedia caches and sbert datasets used in the experiments [here](https://drive.google.com/file/d/13tCAk5BU1vZm9esg1jzdRoliunKqe7dl/view?usp=drivesdk)


### Results Download

You can download results and models predictions from article from this [link](https://drive.google.com/file/d/1) and here the file containing the [Ablation Study](https://drive.google.com/file/d/1hHQVOFatKC6b31LdPJH1bOOgrH6feQaR/view?usp=drivesdk). This way you can audit results and compare to your own experiments.


