"""
This code is adapted from (https://github.com/microsoft/MSMARCO-Passage-Ranking).

"""
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, models, losses, InputExample
import logging
from datetime import datetime
from torch.utils.data import Dataset
import argparse
from sklearn.model_selection import train_test_split
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import InputExample

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--train_file", type=str)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--model_name", default="sentence-transformers/all-mpnet-base-v2")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
args = parser.parse_args()

print(args)

# The  model we want to fine-tune
model_name = args.model_name

train_batch_size = args.train_batch_size           #Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
num_epochs = args.epochs                 # Number of epochs we want to train
train_file = args.train_file

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = './saved-models/sbert-{}-squad-2-0'.format(model_name.replace("/", "-"))


class HardSQUADDataset(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus

    def __getitem__(self, item):
        query_text = self.corpus[item]['question']
        pos_text = self.corpus[item]['pos_sentence']
        neg_text = self.corpus[item]['neg_sentence']

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.corpus)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
with open(train_file, 'r') as f:
    train_data = json.load(f)

train_data, val_data = train_data[:int(len(train_data)*0.9)], train_data[int(len(train_data)*0.9):]

val_examples = [InputExample(texts=[item['question'], item['pos_sentence']], label=1.0) for item in val_data]
val_examples.extend([InputExample(texts=[item['question'], item['neg_sentence']], label=0.0) for item in val_data])


# a dictionary
train_dataset = HardSQUADDataset(corpus=train_data)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
#val_dataset = HardSQUADDataset(corpus=val_data)
#val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=train_batch_size)


train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=EmbeddingSimilarityEvaluator.from_input_examples(val_examples),
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=len(train_dataloader),
          optimizer_params = {'lr': args.lr},
          )

# Save the model
model.save(model_save_path)