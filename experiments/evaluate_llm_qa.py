
from tqdm import tqdm

from collection_index import BM25Index, DenseIndex, reciprocal_rank_fusion, get_sentences_questions
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, BitsAndBytesConfig
import torch
from peft import AutoPeftModelForCausalLM
from setup import SesameConfig
from evaluate_qa import QAEvaluator

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger('transformers').setLevel(logging.ERROR)


class QALLMEvaluator(QAEvaluator):

    MAX_NEW_TOKENS = 32
    USE_INST_TOKEN = False

    def __init__(self, sesame_config: SesameConfig):
        self.sesame_config = sesame_config

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info('loading finetuned model and tokenizer...')


        compute_dtype = getattr(torch, "float16")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        logger.debug(f'loading LLM model {self.sesame_config.finetuned_model_path} and tokenizer')
        # load base LLM model and tokenizer
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.sesame_config.finetuned_model_path,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16, #torch.float16,
            load_in_4bit=True,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.sesame_config.finetuned_model_path, 
                                                       trust_remote_code=True, use_fast=False)

        #self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.resize_token_embeddings(len(self.tokenizer))
        # Gradient checkpointing is used by default but not compatible with caching
        self.model.config.use_cache = False

        self.model.bfloat16()
        self.tokenizer.padding_side = "right"
        #logger.debug(f'Tokenizer padding side: {self.tokenizer.padding_side}')
        #self.model = self.model.merge_and_unload()

    def evaluate(self):

        references = []
        correct_contexts = []
        found_contexts = []
        predictions = []
        predictions_with_correct_context = []


        data = self.load_data()
        logger.info('preprocessing data')        
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


        logger.info('Generating predictions for correct contexts...')
        prompts = [self.sesame_config.prompt_util.get_prompt(example) for example in tqdm(triples)]
        
        pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, 
                        batch_size=8, max_new_tokens=self.MAX_NEW_TOKENS, do_sample=True, 
                        top_p=self.sesame_config.top_p, top_k=self.sesame_config.top_k, 
                        temperature=self.sesame_config.temperature)
        

        outputs = pipe(prompts)
        
        outputs = [self.sesame_config.prompt_util.get_response(out[0]['generated_text']) for out in tqdm(outputs)]
        
        predictions_with_correct_context = [{'id':str(idx), 'prediction_text': pred} for idx, pred in enumerate(outputs)]
        
        if self.sesame_config.do_search:
            logger.info('Generating prompts for best contexts...')
            prompts = [self.sesame_config.prompt_util.get_prompt({'question': example['question'], 'context': context}) for example,context in zip(triples, found_contexts)]
            
            logger.info('Generating predictions for best contexts...')

            outputs = pipe(prompts)
            
            outputs = [self.sesame_config.prompt_util.get_response(out[0]['generated_text']) for out in tqdm(outputs)]
            
            predictions = [{'id':str(idx), 'prediction_text': pred} for idx, pred in enumerate(outputs)]

        return predictions, predictions_with_correct_context, references
    
