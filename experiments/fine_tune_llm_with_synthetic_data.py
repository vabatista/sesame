
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from fine_tune_with_synthetic_data import QAFineTuner
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer, Trainer
from peft import LoraConfig, PeftModel, get_peft_model

from peft.utils import prepare_model_for_kbit_training

from trl import SFTTrainer
import datasets
from setup import SesameConfig

import logging
from torch.utils.data import random_split
import pandas as pd
# wd = Path(__file__).parent.parent.resolve()
# sys.path.append(str(wd))


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QALLMFineTuner(QAFineTuner):
    MAX_LEN = 1024
    DEVICE_MAP = {"": 0}

    def __init__(self, sesame_config: SesameConfig):
        self.sesame_config = sesame_config

        logger.info('loading model and tokenizer')
        # self.tokenizer = Tokenizer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sesame_config.base_model_name, device_map=self.DEVICE_MAP, 
            trust_remote_code=True, use_fast=False)
        
        self.tokenizer.pad_token_id=0 
        
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.tokenizer.padding_side = "right"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.sesame_config.base_model_name,
            #quantization_config=GPTQConfig(
            #    bits=4, disable_exllama=True, tokenizer=self.tokenizer
            #),
            quantization_config=bnb_config,
            device_map=self.DEVICE_MAP,
            trust_remote_code=True,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            use_auth_token=True,
        )

        self.peft_config = LoraConfig(
            lora_alpha=self.sesame_config.lora_config.lora_alpha,
            lora_dropout=self.sesame_config.lora_config.lora_dropout,
            r=self.sesame_config.lora_config.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        # Configure the pad token in the model
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # Gradient checkpointing is used by default but not compatible with caching
        self.model.config.use_cache = False

        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
        self.model = get_peft_model(self.model, self.peft_config)

        logger.info(self.model.print_trainable_parameters())
        # More info: https://github.com/huggingface/transformers/pull/24906
        self.model.config.pretraining_tp = 1

    def get_prompt(self, example) -> str:

        return self.sesame_config.prompt_util.get_prompt(example), example['answers']['text']

  
    def fine_tune(self):

        data = self.load_data()
        texts, queries, answers = self.preprocess(data)
        examples = [{'question': q, 'context': c, 'answers': a}
                    for q, c, a in zip(queries, texts, answers)]

        prompts = map(self.get_prompt, examples)
        input = datasets.Dataset.from_pandas(pd.DataFrame(data=prompts, columns=['text', 'label']))
        
        logger.info('splitting data into train and validation sets')
        input = input.train_test_split(test_size=self.sesame_config.test_split_fraction)

        logger.info('training')
        self.train(input['train'], input['test'], self.sesame_config.epochs)

    def train(self, train_dataset, val_dataset):

        # . For fine-tuning, as far as I could figure out, the learning rate was 5e-8 (no warmup), 1 epoch, max sequence length of 512,
        # and seed 42. These might be the final parameters that we used, but not a 100% sure.
        training_args = TrainingArguments(
            output_dir=self.sesame_config.output_dir,          # output directory
            num_train_epochs=self.sesame_config.epochs,              # total # of training epochs
            #max_steps=10, # FOR DEBUG ONLY
            per_device_train_batch_size=4,  # batch size per device during training
            per_device_eval_batch_size=4,   # batch size for evaluation
            logging_dir='./logs',            # directory for storing logs
            save_steps=10000,
            save_total_limit=2,
            logging_steps=20,
            report_to=None,
            learning_rate=self.sesame_config.lr,
            save_strategy="no",
            evaluation_strategy="epoch",
            warmup_ratio=0.1,
            weight_decay=0.001,
            gradient_accumulation_steps=2,
            optim = "paged_adamw_32bit",
            fp16=True
        )

        if self.mask_inputs:
            logger.debug("Enabling masked inputs")
            response_template = "\n### Response:"
            response_template_ids = self.tokenizer.encode(response_template, add_special_tokens=False)[2:]  
            data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=self.tokenizer)
        else:
            logger.debug("Disabling masked inputs")
            data_collator = None
        

        trainer = SFTTrainer(
            self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=self.MAX_LEN,
            args=training_args,
            peft_config=self.peft_config,
            tokenizer=self.tokenizer,
            packing = False,
            data_collator=data_collator
        )

        

        trainer.train()

        logger.info('saving model')
        self.tokenizer.save_pretrained(self.sesame_config.output_dir)
        trainer.save_model(self.sesame_config.output_dir)

