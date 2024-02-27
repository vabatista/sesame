import json
import random

class SQuADSampler():
    def __init__(self, squad_path) -> None:
       
        self.data = self.__load_squad(squad_path)
        self.samples = self.__get_contexts_questions_answers(self.data)


    def __load_squad(self, squad_path):
        with open(squad_path, 'r') as file:
            json_list = list(file)
        
        data = []
        # [1:] removes the header
        for item in json_list[1:]:
            item = json.loads(item)
            data.append(item)

        return data
    

    def __get_contexts_questions_answers(self, json_data):
        data_list = []

        for paragraph in json_data:
            #context = self.clean_context(paragraph['context'])
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if 'detected_answers' in qa:
                    answer = list(set([a['text'] for a in qa['detected_answers']]))
                else:
                    answer = list(qa['answers'])

                data_dict = {
                    'context': context,
                    'question': question,
                    'answer': answer
                }
                data_list.append(data_dict)

        return data_list    
    
    def get_random_sample(self):
        return random.choice(self.samples)
        


class Prompt():
    
    language = ""
    use_one_shot = None
    def __init__(self, language="en", use_one_shot=True, squad_dataset_path=None) -> None:
        self.language = language
        self.use_one_shot = use_one_shot
        self.sampler = SQuADSampler(squad_dataset_path)

    def get_prompt(self, example) -> str:
          
        if self.language == "en":
                return f"""[INST]<<SYS>> You are a Machine Reading Comprehension System, for extractive Question Answering. 
Below is an instruction that describes a extractive question answering task (like SQuAD). There is a 'Question' and a 'Context'.
Write a response that answers the 'Question' in most short and objective way. Use no more than 20 words. 
Your response must be a span (substring with start_char and end_char) from the given 'Context'. 
If the answer is not in the 'Context', provide the best answer you can.
{self._get_example() if self.use_one_shot else ""} 
<</SYS>>

### Context:\n{example['context']}

### Question:\n{example['question']}

### Response:[/INST]""".strip()

        else:
                return f"""[INST]<<SYS>>Você é um assistente de resposta a questionamentos extrativo. 
Abaixo há uma instrução para você seguir que descreve uma tarefa de questionamento extrativo (como no SQuAD). Há uma pergunta e um contexto.
Escreva uma resposta que responda a pergunta de forma mais curta e objetiva. Use no máximo 20 palavras. 
A resposta deve fazer parte do texto do contexto (uma substring dele). 
Se você achar que a resposta não está no contexto, forneça a melhor resposta que você tem.
{self._get_example() if self.use_one_shot else ""} 
<</SYS>>
### Contexto:\n{example['context']}

### Questão:\n{example['question']}

### Resposta:[/INST]""".strip()                


    def _get_example(self):
        sample = self.sampler.get_random_sample()
        if self.language == "en":
            return f'''
Here is an example

### Context:
{sample['context']}

### Question:
{sample['question']}

### Response:
{sample['answer'][0]}

'''
        elif self.language == "pt":
            return f'''
Aqui está um exemplo

### Contexto:
{sample['context']}

### Questão:
{sample['question']}

### Resposta:
{sample['answer'][0]}

'''
        

    def get_response(self, txt):
        sep = "[/INST]"
        idx = txt.find(sep)
        if idx == -1:
            sep = "### Response:" if self.language == "en" else "### Resposta:"    
            idx = txt.find(sep)
        response = txt.split(sep)[-1].strip()
        if response.find("###") != -1:
            response = response.split("###")[-1].strip()
        if response.find("<<SYS>>") != -1:
            response = response.split("<<SYS>>")[-1].strip()
        if response.find("<<") != -1:
            response = response.split("<<")[-1].strip()
        return response