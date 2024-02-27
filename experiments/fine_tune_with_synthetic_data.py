import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
from transformers import DefaultDataCollator
import pandas as pd
import datasets
from setup import SesameConfig

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


   
class QAFineTuner:
    MAX_LEN = 512
    
    sesame_config = None 

    def __init__(self, sesame_config: SesameConfig):
        self.sesame_config = sesame_config
        logger.info('loading model and tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(self.sesame_config.base_model_name, device_map='auto')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.sesame_config.base_model_name)
        self.model.to(self.device)
        

    def load_data(self):
        logger.info('loading data')
        with open(self.sesame_config.train_file, 'rb') as f:
            data = json.load(f)
        return data[1:] # Skip the first element which is the header

    def preprocess(self, data):
        texts = []
        queries = []
        answers = []

        # Search for each passage, its question and its answer
        for passage in data:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    if answer['answer_start'] > 0:
                        # Store every passage, query and its answer to the lists
                        texts.append(context)
                        queries.append(question)
                        answers.append(answer)
        return texts, queries, answers


    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            
            start_char = answer["answer_start"]
            end_char = answer["answer_start"] + len(answer["text"])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


    def train(self, train_dataset, val_dataset, epochs):

        ##. For fine-tuning, as far as I could figure out, the learning rate was 5e-8 (no warmup), 1 epoch, max sequence length of 512, 
        # and seed 42. These might be the final parameters that we used, but not a 100% sure.
        training_args = TrainingArguments(
            output_dir=self.sesame_config.output_dir,          # output directory
            num_train_epochs=epochs,              # total # of training epochs
            per_device_train_batch_size=32,  # batch size per device during training
            per_device_eval_batch_size=32,   # batch size for evaluation
            logging_dir='./logs',            # directory for storing logs
            save_steps=10000,
            save_total_limit=2,
            logging_steps=50,
            report_to=None,
            learning_rate=self.sesame_config.lr,
            save_strategy = "no",
            evaluation_strategy="epoch",
            gradient_accumulation_steps=4,
            weight_decay = 0,
            lr_scheduler_type = "constant_with_warmup",
            #warmup_ratio=0.1
        )

        trainer = Trainer(
            model=self.model,                         # the instantiated � Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,           # evaluation dataset
            tokenizer=self.tokenizer,
            data_collator = DefaultDataCollator()
        )
        trainer.train()

        logger.info('saving model')
        self.tokenizer.save_pretrained(self.sesame_config.output_dir)
        trainer.save_model(self.sesame_config.output_dir)
        

    def fine_tune(self):

        data = self.load_data()
        texts, queries, answers = self.preprocess(data)
        
        df_data = pd.DataFrame({'id': range(len(texts)), 'context': texts, 'question': queries, 'answers': answers})
        input = datasets.Dataset.from_pandas(df_data)
        #logger.info('finding end positions')
        #train_encodings = self.encode_data(texts, queries, answers)
        logger.info('tokenizing data')
        tokenized_input = input.map(self.preprocess_function, batched=True, batch_size=2, remove_columns=['context', 'question', 'answers'])

        logger.info('splitting data into train and validation sets')
        tokenized_input = tokenized_input.train_test_split(test_size=self.sesame_config.test_split_fraction)

        logger.info('training')
        self.train(tokenized_input['train'], tokenized_input['test'], self.sesame_config.epochs)

       
        
