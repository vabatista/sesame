import re
import sys
import argparse
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BitsAndBytesConfig
import warnings

from nltk.tokenize import sent_tokenize
from peft import AutoPeftModelForCausalLM

from tqdm import tqdm
import json

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# To supress "The model 'PeftModelForCausalLM' is not supported for text-generation."
warnings.filterwarnings("ignore")

class SyntheticDatasetGenerator:
    def __init__(self, t5_model_name, t5_ans_model_name, input_file, output_file):
        self.t5_model_name = t5_model_name
        self.t5_ans_model = t5_ans_model_name
        self.input_file = input_file
        self.output_file = output_file
       

    def _load_model(self):
        logger.info('loading T5 model')
        
        nlp = pipeline("question-generation", 
                    model=self.t5_model_name, 
                    tokenizer=self.t5_model_name,
                    ans_model=self.t5_ans_model,
                    ans_tokenizer=self.t5_ans_model,
                    use_cuda=True)
        nltk.download('punkt')
        logger.info('T5 model loaded - {}'.format(self.t5_model_name))
        return nlp


    def _extract_contexts(self):
        logger.info('extracting contexts from input file')
        with open(self.input_file, 'r') as file:
            json_list = list(file)
        
        contexts = []
        for item in json_list[1:]:
            item = json.loads(item)
            context = item['context']
            #context = re.sub('\[PAR\]|\[TLE\]|\[SEP\]', '', context)  # Remove PAR] [TLE] from context
            #context = re.sub('\n', ' ', context)  # Remove HTML tags from context
            #context = re.sub('<[^<]+?>', ' ', context)  # Remove HTML tags from context
            #context = re.sub('\s{2,}', ' ', context)  # Replace two or more spaces with a single space
            contexts.append(context)
            
        return list(set(contexts))

    def generate_dataset(self):

        contexts = self._extract_contexts()
        nlp = self._load_model()
        synthetic_dataset = [{"header": {"dataset": "synthetic dataset", "split": "train"}}]
        
        total_sentences = 0
        
        for context in tqdm(contexts):
            try:
                sentences = nltk.sent_tokenize(context)
                total_sentences += len(sentences)
            except Exception as ex:
                logger.error('Error tokenizing context: ' + context)
                logger.error(ex)

        avg_sentences = total_sentences // len(contexts)
        avg_sentences = min(15, avg_sentences)
        logger.info('average number of sentences per context: {}'.format(avg_sentences))

        idx = 0
        logger.info('generating questions and answers pairs')
        for context in tqdm(contexts):
            try:
                sentences = nltk.sent_tokenize(context)
                chunks = [sentences[i:i+avg_sentences] for i in range(0, len(sentences), avg_sentences)]
                for chunk in chunks:
                    chunk_context = ' '.join(chunk)
                    
                    gen_qas = nlp(chunk_context)
                    if len(gen_qas)>0:
                        qa_list = []
                        for gqa in gen_qas:
                            qa_list.append({'question': gqa['question'], 'answers': 
                                            [{'text': gqa['answer'], 'answer_start': chunk_context.find(gqa['answer'])}]})
                        synthetic_dataset.append({'context': chunk_context, 'qas': qa_list})
            except Exception as ex:
               logger.error(f'Error generating QA in chunck of size {len(chunk)}: {chunk_context}')
               logger.error(ex)
                
            idx +=1

        with open(self.output_file, "w") as file:
            json.dump(synthetic_dataset, file, indent=2)    





class LLMSyntethicDatasetGenerator(SyntheticDatasetGenerator):

    def __init__(self, model_path, ans_model_name, input_file, output_file):
        self.model_name = model_path
        self.input_file = input_file
        self.output_file = output_file

    
    def _load_model(self):
        ## Load LLama Model for prediction
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = 0.1
        logger.info('loading finetuned model and tokenizer...')

        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        logger.debug(f'loading LLM model {self.model_name} and tokenizer')
        # load base LLM model and tokenizer
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16, #torch.float16,
            load_in_4bit=True,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)


        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.use_cache = False
        self.model.bfloat16()
        self.tokenizer.padding_side = "right"        

    def generate_prompt(self, context) -> str:
        return f"""[INST] <<SYS>>
    Below is context (paragraph). Your task is to create one or more question and answer pairs about the text, respecting these rules:
    1- Your answer should be always a span (substring) of the context
    2- You should also provide a char span of the answer relative to the context
    3- Your output should be in JSON format like this:
    'question': YOUR_QUESTION, 'answer': YOUR_ANSWER, 'answer_start': ANSWER_START
    <</SYS>>
    ### Context:\n{context} 
    [/INST] """.strip()


    def generate_dataset(self):

        contexts = self._extract_contexts()
        self._load_model()
        synthetic_dataset = [{"header": {"dataset": "synthetic dataset", "split": "train"}}]
        
        total_sentences = 0
        
        for context in tqdm(contexts):
            try:
                sentences = nltk.sent_tokenize(context)
                total_sentences += len(sentences)
            except Exception as ex:
                logger.error('Error in context: ' + context)
                logger.error(ex)

        avg_sentences = total_sentences // len(contexts)
        logger.info('average number of sentences per context: {}'.format(avg_sentences))

        idx = 0
        logger.info('generating questions and answers pairs')
        
        pattern = re.compile(r'[Q|q]uestion: (.+?)\n[A|a]nswer: (.+?)\n', re.DOTALL)

        for context in tqdm(contexts):
            sentences = nltk.sent_tokenize(context)
            chunks = [sentences[i:i+avg_sentences] for i in range(0, len(sentences), avg_sentences)]
            for chunk in chunks:
                chunk_context = ' '.join(chunk)
                
                prompt = self.generate_prompt(chunk_context)
                pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, 
                                batch_size=4, max_new_tokens=128, do_sample=True, 
                                top_p=0.8, top_k=100, temperature=0.2)
                

                output = pipe(prompt)
                
                gen_qas = output[0]['generated_text'].split('[/INST]')[1].strip()

                # Define the regex pattern for extracting question and answer pairs
                

                # Find all matches in the text
                matches = pattern.findall(gen_qas)
                qa_list = []
                # Print the extracted pairs
                for match in matches:
                    qa_list.append({'question': match[0], 'answers': 
                                         [{'text': match[1], 'answer_start': chunk_context.find(match[1])}]})
                if len(qa_list)>0:
                    synthetic_dataset.append({'context': chunk_context, 'qas': qa_list})
                
            idx +=1

        with open(self.output_file, "w") as file:
            json.dump(synthetic_dataset, file, indent=2)    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate synthetic dataset from URL')

    parser.add_argument('--t5_model_name', type=str, help='path or huggingface question generation model name')
    parser.add_argument('--t5_ans_model_name', type=str, help='path or huggingface answer generation model name')
    parser.add_argument('--input_file', type=str, help='Json with squad format')
    parser.add_argument('--output_file', type=str, help='Path to save the generated dataset')
    parser.add_argument('--llm', action='store_true', help='Enable LLM (LLaMa 2) prediction')

    #parser.add_argument('--num_sentences', type=int, help='Number of sentences per chunk')
    args = parser.parse_args()
    if args.llm:
        generator = LLMSyntethicDatasetGenerator(args.t5_model_name, args.t5_ans_model_name, args.input_file, args.output_file)
        from transformers import pipeline
    else:
        sys.path.insert(0,"..")
        sys.path.insert(0,"./pretraining/question_generation")
        from pipelines import pipeline
        generator = SyntheticDatasetGenerator(args.t5_model_name, args.t5_ans_model_name, args.input_file, args.output_file)
    generator.generate_dataset()