import torch
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lightning.fabric.plugins import BitsandbytesPrecision
from lit_gpt.utils import get_default_supported_precision
from prompt import Prompt



class LoraConfig:
    def __init__(self, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_query=True,
                 lora_key=False, lora_value=True, lora_projection=False, lora_mlp=False, lora_head=False):
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_query = lora_query
        self.lora_key = lora_key
        self.lora_value = lora_value
        self.lora_projection = lora_projection
        self.lora_mlp = lora_mlp
        self.lora_head = lora_head

class SesameConfig:

    def __init__(self, base_model_name=None, finetuned_model_path=None, similarity_model_path=None, 
                 inference_file=None, train_file=None, output_dir=None, ranking_file=None, do_search=True,
                 contents_file=None, quantize=False, lr=None, mask_inputs=False, language=None, one_shot=None,
                 epochs=1, top_k_contexts=None, temperature=0.1, top_p=0.9, top_k=200, test_split_fraction=0.2):
        self.base_model_name = base_model_name
        self.finetuned_model_path = finetuned_model_path
        self.similarity_model_path = similarity_model_path
        self.inference_file = inference_file
        self.train_file = train_file
        self.output_dir = output_dir
        self.ranking_file = ranking_file
        self.contents_file = contents_file
        self.quantize = quantize
        self.lr = lr
        self.mask_inputs = mask_inputs
        self.epochs = epochs
        self.top_k_contexts = top_k_contexts
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.test_split_fraction = test_split_fraction
        self.do_search = do_search

        if language == "en":
            self.prompt_util = Prompt(language=language, use_one_shot=one_shot, squad_dataset_path='qa-datasets/originals/SQuAD.jsonl')
        elif language == "pt":
            self.prompt_util = Prompt(language=language, use_one_shot=one_shot, squad_dataset_path='qa-datasets/originals/squad-ptbr-dev-v1.1.jsonl')

        self.lora_config = LoraConfig(lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_query=True,
                            lora_key=True, lora_value=True, lora_projection=True, lora_mlp=True, lora_head=True)
        
        if self.quantize:
            #quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
            #dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            # Uncomment to use bitsandbytes precision
            self.qlora_config = BitsandbytesPrecision('nf4', torch.bfloat16)
            self.precision = None
        else:
            self.precision = get_default_supported_precision(training=True)
            self.qlora_config = None
            