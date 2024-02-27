import json
import argparse
import random
from setup import SesameConfig

from datasets import load_metric
import os
import uuid

from nltk.corpus import stopwords

from evaluate_qa import QAEvaluator, WikiEvaluator
from evaluate_llm_qa import QALLMEvaluator
from evaluate_llm_litgpt import QALLMLitGPTEvaluator, LLMWikiEvaluator

from fine_tune_with_synthetic_data import QAFineTuner
from fine_tune_llm_litgpt import QALLMFineTunerLit
from fine_tune_llm_with_synthetic_data import QALLMFineTuner


import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('full_experiment')

#nltk.download('stopwords')
#st = stopwords.words('portuguese')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune BERT for question answering')

    parser.add_argument('--do_train', action='store_true', help='Enable training', default=False)
    parser.add_argument('--do_eval', action='store_true', help='Enable evaluation', default=False)    
    parser.add_argument('--do_search', action='store_true', help='Enable search for best contexts in evaluation', default=False)

    parser.add_argument('--language', type=str, help='Language of the input data', default='en')
    parser.add_argument('--use_one_shot', action='store_true', help='Use one-shot learning', default=True)

    ## Finetuning args
    parser.add_argument('--base_model_name', type=str, help='Name or path of the pre-trained BERT model to use')
    parser.add_argument('--train_input_file', type=str, help='Path to the synthetic file in SQuAD format')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate to fine-tuned model')
    parser.add_argument('--mask_inputs', action='store_true', help="Whenever to mask the prompt in the labels or not when using LLMs")

    ## Evaluation args
    parser.add_argument('--finetuned_model_path', type=str, help='Path to the fine-tuned model')
    parser.add_argument('--similarity_model_path', type=str, help='Name or path of the pretrained sentence-bert model to use')
    parser.add_argument('--input_file', type=str, help='Path to the SQuAD like data file')
    parser.add_argument('--n_samples', type=int, default=0, help='Number of samples to evaluate on. 0 takes all samples')
    parser.add_argument('--top_k_contexts', type=int, default=4, help='Number of top-k sentences to retrieve')
    parser.add_argument('--test_split', type=float, help='percentage of data to calculate validation during training')
    parser.add_argument('--results_path', type=str, help='Path to save the results to', default='./results')
    
    ## LLM specific args
    parser.add_argument('--llm', action='store_true', help='Enable LLM (LLaMa 2) finetuning')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature for generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p cummulative prob for generation')
    parser.add_argument('--top_k', type=int, default=200, help='top_k tokens to sample for generation')
    parser.add_argument('--use_hf_ft', action='store_true', help='Use huggingface fine-tuning instead of lit-gpt fine-tuning', default=False)
    parser.add_argument('--quantize', action='store_true', help='Quantize the model', default=False)

    # When running a wiki full search experiment
    parser.add_argument('--wiki', action='store_true', help='Use stored wiki cache instead of search', default=False)
    parser.add_argument('--ranking_file', type=str, help='Name or path of rankings file for each query', required=False)
    parser.add_argument('--contents_file', type=str, help='Name or path of wikipedia contents file', required=False)

    args = parser.parse_args()

    myuuid = uuid.uuid4()
    
    ## Create a folder for all the results for this experiment
    results_folder = os.path.join(args.results_path, 'experiments' ,str(myuuid))
    os.mkdir(results_folder)    

    sesame_config = SesameConfig(base_model_name=args.base_model_name, 
                                 finetuned_model_path=args.train_input_file, 
                                 similarity_model_path=args.similarity_model_path,
                                 inference_file=args.input_file,
                                 train_file=args.train_input_file,
                                 output_dir=results_folder,
                                 epochs=args.epochs,
                                 top_k_contexts=args.top_k_contexts,
                                 temperature=args.temperature,
                                 top_p=args.top_p,
                                 top_k=args.top_k,
                                 test_split_fraction=args.test_split,
                                 lr=args.lr,
                                 mask_inputs=args.mask_inputs,
                                 quantize=args.quantize,
                                 language=args.language,
                                 one_shot=args.use_one_shot,
                                 ranking_file=args.ranking_file,
                                 contents_file=args.contents_file,
                                 do_search=args.do_search)



    
    fh = logging.FileHandler(os.path.join(results_folder, 'experiment.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info(' ========  Experiment UUID:' + str(myuuid) + " ========\n\n")

    ## Save the experiment args to experiments.csv
    with open(os.path.join(args.results_path, 'experiments.csv'), 'a') as f:
        f.write(f'{myuuid}, {args.train_input_file}, {args.input_file}, {args.base_model_name}, {args.epochs}, \
                {args.top_k_contexts}, {args.test_split}, {args.lr}, {args.temperature}, {args.top_p}, {args.top_k}, {args.mask_inputs}\n')
    
    if args.do_train:
        if args.llm:
            logger.info('===== Fine-tuning step with LLM =====')
            if args.use_hf_ft:
                fine_tuner = QALLMFineTuner(sesame_config)
            else:
                fine_tuner = QALLMFineTunerLit(sesame_config)
        else:
            logger.info('===== Fine-tuning step with BERT =====')
            fine_tuner = QAFineTuner(sesame_config)

        fine_tuner.fine_tune()

        qa_model_path = os.path.join(results_folder, 'experiments', 'saved-models')
        sesame_config.finetuned_model_path = qa_model_path

    if args.do_eval:
        logger.info('===== Evaluation step =====')
        if args.llm:
            if args.use_hf_ft:
                evaluator = QALLMEvaluator(sesame_config)
            else:
                if args.wiki:
                    evaluator = LLMWikiEvaluator(sesame_config)
                else:
                    evaluator = QALLMLitGPTEvaluator(sesame_config)
        else:
            if args.wiki:
                evaluator = WikiEvaluator(sesame_config)
            else:
                evaluator = QAEvaluator(sesame_config)

        predictions, predictions_with_correct_context, references = evaluator.evaluate()

        predictions_file = os.path.join(results_folder, 'predictions.json')
        predictions_with_correct_context_file = os.path.join(results_folder, 'predictions_with_correct_context.json')
        references_file = os.path.join(results_folder, 'references.json')

        with open(predictions_file, 'w') as f:
            json.dump(predictions, f)

        with open(predictions_with_correct_context_file, 'w') as f:
            json.dump(predictions_with_correct_context, f)

        with open(references_file, 'w') as f:
            json.dump(references, f)

        squad_metric = load_metric('squad')
        squad_metric_results = {}

        if len(predictions) == len(references):
            squad_metric_results['predictions'] = squad_metric.compute(predictions=predictions, references=references)
            squad_metric_results['predictions']['precision'] = evaluator.calc_precision(predictions=predictions, references=references)        
        if len(predictions_with_correct_context) == len(references):
            squad_metric_results['predictions_with_correct_context'] = squad_metric.compute(predictions=predictions_with_correct_context, references=references)
            squad_metric_results['predictions_with_correct_context']['precision'] = evaluator.calc_precision(predictions=predictions_with_correct_context, references=references)


        if len(predictions_with_correct_context) > 10:
            ids = random.sample(range(len(predictions_with_correct_context)), 10)
            print('Predictions\n',  [predictions_with_correct_context[i] for i in ids], '\nReferences:\n', 
                [{'id': references[i]['id'], 'text': references[i]['answers']['text']} for i in ids])
        else:
            ids = random.sample(range(len(predictions)), 10)
            print('Predictions\n',  [predictions[i] for i in ids], '\nReferences:\n', 
                [{'id': references[i]['id'], 'text': references[i]['answers']['text']} for i in ids])   
                
        logger.info(f'Squad results {squad_metric_results}')
        
        squad_metric_file = os.path.join(results_folder, 'squad_metric.json')
        with open(squad_metric_file, 'w') as f:
            json.dump(squad_metric_results, f, indent=2)