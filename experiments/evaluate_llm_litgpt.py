import argparse
import random
import re, time, json, os, sys
from tqdm import tqdm
from pathlib import Path
from typing import Literal, Optional

from setup import SesameConfig

from collection_index import BM25Index, DenseIndex, reciprocal_rank_fusion, get_sentences_questions

from datasets import load_metric
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Block, Config, merge_lora_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, gptq_quantization, lazy_load
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy

torch.set_float32_matmul_precision("high")

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from evaluate_qa import QAEvaluator

class QALLMLitGPTEvaluator(QAEvaluator):


    MAX_NEW_TOKENS = 64

    def __init__(self, sesame_config: SesameConfig):
        self.sesame_config = sesame_config

    def load_model(self):
        
        checkpoint_dir = Path(self.sesame_config.base_model_name)

        strategy = "auto"
        devices  = 1

        fabric = L.Fabric(devices=devices, precision=self.sesame_config.precision, 
                          strategy=strategy, plugins=self.sesame_config.qlora_config)
        fabric.launch()

        check_valid_checkpoint_dir(checkpoint_dir)

        config = Config.from_json(
            checkpoint_dir / "lit_config.json",
            r=self.sesame_config.lora_config.lora_r,
            alpha=self.sesame_config.lora_config.lora_alpha,
            dropout=self.sesame_config.lora_config.lora_dropout,
            to_query=self.sesame_config.lora_config.lora_query,
            to_key=self.sesame_config.lora_config.lora_key,
            to_value=self.sesame_config.lora_config.lora_value,
            to_projection=self.sesame_config.lora_config.lora_projection,
            to_mlp=self.sesame_config.lora_config.lora_mlp,
            to_head=self.sesame_config.lora_config.lora_head,
        )

        model_file = "lit_model.pth"
        checkpoint_path = checkpoint_dir / model_file

        logger.info(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
        t0 = time.perf_counter()
        with fabric.init_module(empty_init=True), gptq_quantization(False):
            model = GPT(config)
        logger.info(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        checkpoint = lazy_load(checkpoint_path)

        
        t0 = time.perf_counter()
        ## check if lora_path file exists
        if self.sesame_config.finetuned_model_path != None:
            lora_path = Path(os.path.join(self.sesame_config.finetuned_model_path,'lit_model_lora_finetuned.pth'))
            logger.info(f"Loading LORA model {str(lora_path)!r}")
            lora_checkpoint = lazy_load(lora_path)
            checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
            model.load_state_dict(checkpoint)
            logger.info(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.")
            merge_lora_weights(model)
        else:
            #model.load_state_dict(checkpoint)
            logger.info(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.")
        model.eval()
        model = fabric.setup(model)

        tokenizer = Tokenizer(checkpoint_dir)
        return model, tokenizer, fabric

    def evaluate(self):
        
        references = []
        correct_contexts = []
        found_contexts = []
        predictions = []
        predictions_with_correct_context = []


        model, tokenizer, fabric = self.load_model()

        logger.info('preprocessing data')        
        data = self.load_data()
        sentences, questions, q2s = get_sentences_questions(data)

        logger.info('creating bm25 index')
        bm25index = BM25Index(sentences)

        logger.info('creating dense index')
        denseindex = DenseIndex(sentences, self.sesame_config.similarity_model_path)

        triples = self.get_contexts_questions_answers(data)

        for idx, triple in enumerate(triples):
            question, answers, context = triple['question'], triple['answer'], triple['context']
            correct_contexts.append(context)
            references.append({'id': str(idx), 'answers': {'answer_start': [context.find(answer) for answer in answers], 
                                                           'text': [answer for answer in answers]}})

        if self.sesame_config.do_search:
            logger.info('Searching for best contexts...')
            for idx, triple in enumerate(tqdm(triples)):

                text_hits = bm25index.search(triple['question'], self.sesame_config.top_k_contexts)
                dense_hits = denseindex.search(triple['question'], self.sesame_config.top_k_contexts)
                hits = reciprocal_rank_fusion(dense_hits, text_hits)

                top_k_sentences = ''
                for hit in hits[0:self.sesame_config.top_k_contexts]:
                    top_k_sentences += self.get_sentence_window(sentences, hit, window=self.window)
                found_contexts.append(top_k_sentences)

        

        logger.info('Generating predictions...')
        t0 = time.perf_counter()
        
        for idx, triple in enumerate(tqdm(triples)):
            question, answers, context = triple['question'], triple['answer'], triple['context']

            sample = {"question": question, "context": context}
            prompt = self.sesame_config.prompt_util.get_prompt(sample)
            encoded = tokenizer.encode(prompt, device=fabric.device)
            prompt_length = encoded.size(0)
            max_returned_tokens = prompt_length + self.MAX_NEW_TOKENS
            try:        
                with fabric.init_tensor():
                    # set the max_seq_length to limit the memory usage to what we need
                    model.max_seq_length = max_returned_tokens
                    # enable the kv cache
                    model.set_kv_cache(batch_size=1)

                y = self.generate(model, encoded, max_returned_tokens, temperature=self.sesame_config.temperature, 
                                  top_k=self.sesame_config.top_k, eos_id=tokenizer.eos_id)

                output = tokenizer.decode(y)
                if (idx % 100 == 0):
                    logger.debug(f'===========  Generated output: {output}')
                
                output = self.sesame_config.prompt_util.get_response(output)
                if (idx % 100 == 0):
                    logger.debug(f'===========  Generated response: {output}')
                    logger.debug('\n\n')
                    
            except:
                output = ''

            predictions_with_correct_context.append({'id': str(idx), 'prediction_text':  output})
        
            if self.sesame_config.do_search:
                sample = {"question": question, "context": found_contexts[idx]}
                prompt = self.sesame_config.prompt_util.get_prompt(sample)
                encoded = tokenizer.encode(prompt, device=fabric.device)
                prompt_length = encoded.size(0)
                max_returned_tokens = prompt_length + self.MAX_NEW_TOKENS
                try:        
                    with fabric.init_tensor():
                        # set the max_seq_length to limit the memory usage to what we need
                        model.max_seq_length = max_returned_tokens
                        # enable the kv cache
                        model.set_kv_cache(batch_size=1)

                    y = self.generate(model, encoded, max_returned_tokens, temperature=self.sesame_config.temperature, 
                                      top_k=self.sesame_config.top_k, eos_id=tokenizer.eos_id)

                    output = tokenizer.decode(y)
                    output = self.sesame_config.prompt_util.get_response(output)
                except:
                    output = ''

                predictions.append({'id': str(idx), 'prediction_text':  output})


        t = time.perf_counter() - t0
        logger.info(f"\n\nTime for inference: {t:.02f} sec total")
        #logger.info(f'Number of errors generating answers: {n_errors}')

        if fabric.device.type == "cuda":
            logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        
        return predictions, predictions_with_correct_context, references
   

    @torch.inference_mode()
    def generate(self,
        model: GPT,
        idx: torch.Tensor,
        max_returned_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

        The implementation of this function is modified from A. Karpathy's nanoGPT.

        Args:
            model: The model to use.
            idx: Tensor of shape (T) with indices of the prompt sequence.
            max_returned_tokens: The maximum number of tokens to return (given plus generated).
            temperature: Scales the predicted logits by 1 / temperature.
            top_k: If specified, only sample among the tokens with the k highest probabilities.
            eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        """
        T = idx.size(0)
        assert max_returned_tokens > T
        if model.max_seq_length < max_returned_tokens - 1:
            # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
            # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
            # not support it to avoid negatively impacting the overall speed
            raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

        device, dtype = idx.device, idx.dtype
        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
        empty[:T] = idx
        idx = empty
        input_pos = torch.arange(0, T, device=device)

        # generate up to a fixed number of tokens
        for _ in range(max_returned_tokens - T):
            x = idx.index_select(0, input_pos).view(1, -1)

            # forward
            logits = model(x, input_pos)
            logits = logits[0, -1] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

            # advance
            input_pos = input_pos[-1:] + 1

            # concatenate the new generation
            idx = idx.index_copy(0, input_pos, idx_next)

            # if <eos> token is triggered, return the output (stop generation)
            if idx_next == eos_id:
                return idx[:input_pos]  # include the EOS token

        return idx
        

'''
This class is used to evaluate a fine-tuned BERT model on a dataset based on Wikipedia.
I tested only with PopQA and TriviaQA. Original datasets provided a context for each question based on
BM25 results/snipets. 
I used a different ranking model to retrieve top 100 passages from castorini/odqa-wiki-corpora on huggingface
It was necessary to search offline and store results into cache because the index is too large.
'''
class LLMWikiEvaluator(QALLMLitGPTEvaluator):
    
    content_dict = {}
    ranks = {}


    def __init__(self, sesame_config: SesameConfig):
        self.sesame_config = sesame_config
    
    '''
    For PopQA and TriviaQA only pairs of question and answers are provided.
    '''
    def load_data(self):

        logger.info('loading data')
        with open(self.sesame_config.inference_file, 'r') as file:
            json_list = list(file)
        
        data = []
        # [1:] removes the header, but it is not necessary for PopQA and TriviaQA
        # But I made a mistake creating the cache from the 2nd line, so I need to remove the first to match indexes
        for item in json_list: #[1:]:
            item = json.loads(item)
            data.append(item)

        logger.info(f"Total samples loaded: {len(data)}")
        return data

    def load_contents(self):
        logger.info('loading contents index')
        with open(self.sesame_config.contents_file, 'r') as file:
            contents = json.load(file)
        return contents
    
    def load_ranks(self):
        logger.info('loading ranks index')
        with open(self.sesame_config.ranking_file, 'r') as file:
            ranks = json.load(file)
        return ranks

    def evaluate(self):
        
        references = []
        found_contexts = []
        predictions = []
        questions = []

        data = self.load_data()
        self.content_dict = self.load_contents()
        self.ranks = self.load_ranks()

        not_found_count = 0
        for idx, qa in tqdm(enumerate(data), total=len(data)):
            question, answers = qa['question'], qa['answer']
            if type(answers) != list:
                answers = eval(answers)
            if str(idx) not in self.ranks:
                not_found_count += 1
                continue
            questions.append(question)

            found = []
            for i in range(self.sesame_config.top_k_contexts):
                if type(self.ranks[str(idx)][0]) == str:
                    found.append(self.content_dict[self.ranks[str(idx)][i]])
                else:
                    found.append(self.content_dict[self.ranks[str(idx)][i]['id']])
            found_contexts.append(' '.join(found))
            
            references.append({'id': str(idx), 'answers': {'answer_start': [0]*len(answers), 
                                                           'text': [answer for answer in answers]}})

        model, tokenizer, fabric = self.load_model()

        logger.info('Generating predictions for best contexts...')

        for idx, (question, context) in tqdm(enumerate(zip(questions, found_contexts)), total=len(questions)):
            sample = {"question": question, "context": context}
            prompt = self.sesame_config.prompt_util.get_prompt(sample)
            encoded = tokenizer.encode(prompt, device=fabric.device)
            prompt_length = encoded.size(0)
            max_returned_tokens = prompt_length + self.MAX_NEW_TOKENS
            try:        
                with fabric.init_tensor():
                    # set the max_seq_length to limit the memory usage to what we need
                    model.max_seq_length = max_returned_tokens
                    # enable the kv cache
                    model.set_kv_cache(batch_size=1)

                y = self.generate(model, encoded, max_returned_tokens, temperature=self.sesame_config.temperature, 
                                  top_k=self.sesame_config.top_k, eos_id=tokenizer.eos_id)

                
                output = tokenizer.decode(y)
                if (idx % 100 == 0):
                    logger.debug(f'===========  Generated output: {output}')
                
                output = self.sesame_config.prompt_util.get_response(output)
                if (idx % 100 == 0):
                    logger.debug(f'===========  Generated response: {output}')
                    logger.debug(f'===========  Correct Answer: {references[idx]["answers"]["text"]}')  
                
            except:
                output = ''

            predictions.append({'id': str(idx), 'prediction_text':  output})
       
        logger.debug(f'Not found ranks for {not_found_count} questions')

        return predictions, [], references
