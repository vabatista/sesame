# This code was adapted from lit-gpt repo: https://github.com/Lightning-AI/lit-gpt/
import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from setup import SesameConfig

from lit_gpt.lora import GPT, Config, lora_filter, mark_only_lora_as_trainable
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    load_checkpoint,
    num_parameters,
)


from fine_tune_with_synthetic_data import QAFineTuner
import logging
from torch.utils.data import random_split

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainConfig:
    def __init__(self, eval_interval=100, save_interval=100, eval_iters=100, eval_max_new_tokens=100,
                 log_interval=1, devices=1, batch_size=64, micro_batch_size=4,
                 weight_decay=0.01):
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.eval_iters = eval_iters
        self.eval_max_new_tokens = eval_max_new_tokens
        self.log_interval = log_interval
        self.devices = devices
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_iters = batch_size // micro_batch_size
        self.weight_decay = weight_decay
        self.max_iters = 1
        self.warmup_steps = 1



class QALLMFineTunerLit(QAFineTuner):
    

    def __init__(self, sesame_config: SesameConfig):
        self.sesame_config = sesame_config

        torch.set_float32_matmul_precision("high")
        self.checkpoint_dir = Path(self.sesame_config.base_model_name)
        check_valid_checkpoint_dir(self.checkpoint_dir)


        train_config = TrainConfig(eval_interval=1000, save_interval=1000, eval_iters=1000, eval_max_new_tokens=32,
                            log_interval=20, devices=1, batch_size=8, micro_batch_size=1,
                            weight_decay=1e-2)
        
        self.gradient_accumulation_iters = train_config.batch_size // train_config.micro_batch_size
        assert self.gradient_accumulation_iters > 0

        hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

        strategy = "auto"

        fabric = L.Fabric(devices=train_config.devices, strategy=strategy, precision=self.sesame_config.precision, 
                          loggers=logger, plugins=self.sesame_config.qlora_config)
        fabric.print(hparams)
        self.fabric = fabric
        self.train_config = train_config


    def prepare_sample(self, example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool, ignore_index: int) -> None:
        """Processes a single sample.

        Each sample in the dataset consists of:
        - instruction: A string describing the task
        - input: A string holding a special input value for the instruction.
            This only applies to some samples, and in others this is empty.
        - output: The response string

        This function processes this data to produce a prompt text and a label for
        supervised training. The prompt text is formed as a single message including both
        the instruction and the input. The label/target is the same message but with the
        response attached.

        Finally, both the prompt and the label get tokenized. If desired, all tokens
        in the label that correspond to the original input prompt get masked out (default).
        """
        full_prompt = self.sesame_config.prompt_util.get_prompt(example)
        full_prompt_and_response = full_prompt + example["answer"]['text']
        

        encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
        encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_full_prompt_and_response.clone()
        if mask_inputs:
            labels[: len(encoded_full_prompt)] = ignore_index

        return {
            **example,
            "input_ids": encoded_full_prompt_and_response,
            "input_ids_no_response": encoded_full_prompt,
            "labels": labels,
        }        
        
    def create_dataset(self, triples, test_split_fraction=0.1):

        with open(self.checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]


        logger.info("Loading tokenizer...")
        self.tokenizer = Tokenizer(self.checkpoint_dir)

        # Partition the dataset into train and test
        train_set, test_set = random_split(
            triples, [1.0 - test_split_fraction, test_split_fraction], generator=torch.Generator().manual_seed(5)
        )
        train_set, test_set = list(train_set), list(test_set)

        ### For testing purposes. I'll setup test to 0.99 to not train the model more than really few examples.        
        if self.sesame_config.test_split_fraction==0.99:
            test_set = test_set[:50]

        logger.info(f"train has {len(train_set):,} samples")
        logger.info(f"test has {len(test_set):,} samples")

        logger.info("Processing train split ...")
        train_set = [
            self.prepare_sample(
                example=sample,
                tokenizer=self.tokenizer,
                max_length=max_seq_length,
                mask_inputs=self.sesame_config.mask_inputs,
                ignore_index=-1,
            )
            for sample in tqdm(train_set)
        ]
        

        logger.info("Processing test split ...")
        test_set = [
            self.prepare_sample(
                example=sample,
                tokenizer=self.tokenizer,
                max_length=max_seq_length,
                mask_inputs=self.sesame_config.mask_inputs,
                ignore_index=-1,
            )
            for sample in tqdm(test_set)
        ]
        
        return train_set, test_set


    def fine_tune(self):

        data = self.load_data()
        texts, queries, answers = self.preprocess(data)
        examples = [{'question': q, 'context': c, 'answer': a}
                    for q, c, a in zip(queries, texts, answers)]


        train_data, test_data = self.create_dataset(examples, test_split_fraction=self.sesame_config.test_split_fraction)

        self.train_config.max_iters = 1 + self.sesame_config.epochs * len(train_data) // self.train_config.batch_size 
        self.train_config.warmup_steps = int(1 + self.train_config.max_iters*.1) 

        if not any((self.sesame_config.lora_config.lora_query, self.sesame_config.lora_config.lora_key, 
                    self.sesame_config.lora_config.lora_value, self.sesame_config.lora_config.lora_projection, 
                    self.sesame_config.lora_config.lora_mlp, self.sesame_config.lora_config.lora_head)):
            logger.info("Warning: all LoRA layers are disabled!")
        
        configLora = Config.from_name(
            name=self.checkpoint_dir.name,
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
        checkpoint_path = self.checkpoint_dir / "lit_model.pth"

        logger.info(f"Loading model {str(checkpoint_path)!r} with {configLora.__dict__}")
        with self.fabric.init_module(empty_init=(self.train_config.devices > 1)):
            model = GPT(configLora)
        mark_only_lora_as_trainable(model)

        logger.info(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
        logger.info(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

        model = self.fabric.setup_module(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if isinstance(self.fabric.strategy.precision, BitsandbytesPrecision):
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdamW(trainable_params, lr=self.sesame_config.lr, weight_decay=self.train_config.weight_decay)
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=self.sesame_config.lr, weight_decay=self.train_config.weight_decay)
        optimizer = self.fabric.setup_optimizers(optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_config.max_iters // self.train_config.batch_size)

        # strict=False because missing keys due to LoRA weights not contained in state dict
        load_checkpoint(self.fabric, model, checkpoint_path, strict=False)

        #self.fabric.seed_everything(1337 + self.fabric.global_rank)

        train_time = time.perf_counter()

        longest_seq_length, longest_seq_ix = self.get_longest_seq_length(train_data)
        longest_val_seq_length, longest_val_seq_ix = self.get_longest_seq_length(test_data)
        model.max_seq_length = max(longest_seq_length, longest_val_seq_length)
        logger.info(
            f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is \
            {model.max_seq_length} and context length is {model.config.block_size}"
        )

        step_count = 0
        total_lengths = 0
        total_t0 = time.perf_counter()
        
        
        logger.info(f"epochs: {self.sesame_config.epochs}, max_iters: {self.train_config.max_iters}, \
                    warmup_steps: {self.train_config.warmup_steps}")

        for iter_num in tqdm(range(self.train_config.max_iters)):
            if step_count <= self.train_config.warmup_steps:
                # linear warmup
                lr = self.sesame_config.lr * step_count / self.train_config.warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            iter_t0 = time.perf_counter()

            input_ids, targets = self.get_batch(train_data, longest_seq_ix if iter_num == 0 else None)

            is_accumulating = (iter_num + 1) % self.gradient_accumulation_iters != 0
            with self.fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(input_ids, lm_head_chunk_size=128)
                # shift the targets such that output n predicts token n+1
                logits[-1] = logits[-1][..., :-1, :]
                loss = chunked_cross_entropy(logits, targets[..., 1:])
                self.fabric.backward(loss / self.gradient_accumulation_iters)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                if step_count > self.train_config.warmup_steps:
                    scheduler.step()
                step_count += 1

            t1 = time.perf_counter()
            total_lengths += input_ids.size(1)

            if iter_num % self.train_config.log_interval == 0:
                logger.info(
                    f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time: \
                    {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                )

            if not is_accumulating and step_count % self.train_config.eval_interval == 0:
                t0 = time.perf_counter()
                val_loss = self.validate(model, test_data, self.tokenizer)
                t1 = time.perf_counter() - t0
                logger.info(f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
                self.fabric.barrier()
            if not is_accumulating and step_count % self.train_config.save_interval == 0:
                checkpoint_path = os.path.join(self.sesame_config.output_dir , f"iter-{iter_num:06d}-ckpt.pth")
                self.save_lora_checkpoint(model, checkpoint_path)

        
        logger.info(f"Training time: {(time.perf_counter()-train_time):.2f}s")
        if self.fabric.device.type == "cuda":
            logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

        # Save the final LoRA checkpoint at the end of training
        save_path = os.path.join(self.sesame_config.output_dir, "experiments", "saved-models" ,"lit_model_lora_finetuned.pth")
        self.save_lora_checkpoint( model, save_path)


    @torch.inference_mode()
    def validate(self, model: GPT, val_data: List[Dict], tokenizer: Tokenizer) -> torch.Tensor:
        self.fabric.print("Validating ...")
        model.eval()
        losses = torch.zeros(self.train_config.eval_iters)
        for k in range(self.train_config.eval_iters):
            input_ids, targets = self.get_batch(val_data)
            logits = model(input_ids)
            losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
        val_loss = losses.mean()

        # produce an example:
        context = "(CNN) -- What could be more powerful than the tears of a Native American Indian?\n\n\n\nWax on, wax off: Does it make you want to save the rainforests?\n\n\n\nIron Eyes Cody was the face of the Keep American Beautiful campaign of 1971 whose tears marked the plight of the environment, but more importantly kept the problems of pollution in the minds of millions.\n\n\n\nFrom teary Native Americans to witty skits or doom-ladened eco-horror scenarios, the environmental campaign video then has long been a powerful tool for environmental groups to spread their message and raise pubic attention.\n\n\n\nThe rise of YouTube and other video sharing web sites has now meant that individuals can broadcast their own eco-awareness messages and form their own social action networks.\n\n\n\nBut what makes a good video and how much impact do they have? Is it better to be funny or shocking? When you see Harrison Ford getting his chest waxed, do you immediately think about saving the rainforests?\n\n\n\nOr does the sight of celebrity pontificating about the plight of the environment make you want to watch their next film rather calculate your carbon footprint.\n\n\n\nWe've featured three different videos that we like and want to know which ones you think are the best.  Watch the featured videos \u00bb\n\n\n\nLet us know which eco videos have got you going by using the Sound Off box below. Or, e-mail us at ecosolutions@cnn.com.\n\n\n\nWe also want to feature your own environmental videos here on CNN's Eco Solutions. Use the iReport form to send in your film and you could find your environmental efforts make even more impact than Harrison Ford's chest."
        question = "What does the Harrison Ford video feature?"
        prompt = self.sesame_config.prompt_util.get_prompt({'context': context, 'question': question})
        logger.debug(prompt)
        encoded = tokenizer.encode(prompt, device=self.fabric.device)
        with self.fabric.init_tensor():
            # do not set `max_seq_length=max_returned_token` because memory is not a concern here
            model.set_kv_cache(batch_size=1)
        output = self.generate(model, encoded, max_returned_tokens=len(encoded) + self.train_config.eval_max_new_tokens, 
                               temperature=self.sesame_config.temperature)
        model.clear_kv_cache()
        output = tokenizer.decode(output)
        logger.debug(output)

        model.train()
        return val_loss

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

    def save_lora_checkpoint(self, model: torch.nn.Module, file_path: Path) -> None:
        logger.info(f"Saving LoRA weights to {str(file_path)!r}")
        self.fabric.save(file_path, {"model": model}, filter={"model": lora_filter})

    def get_batch(self, data: List[Dict], longest_seq_ix: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(data), (self.train_config.micro_batch_size,))
        if longest_seq_ix is not None:
            # force the longest sample at the beginning so potential OOMs happen right away
            ix[0] = longest_seq_ix

        input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
        labels = [data[i]["labels"].type(torch.int64) for i in ix]

        # this could be `longest_seq_length` to have a fixed size for all batches
        max_len = max(len(s) for s in input_ids)

        def pad_right(x, pad_id):
            # pad right based on the longest sequence
            n = max_len - len(x)
            return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

        x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
        y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

        if self.fabric.device.type == "cuda" and x.device.type == "cpu":
            x, y = self.fabric.to_device((x.pin_memory(), y.pin_memory()))
        else:
            x, y = self.fabric.to_device((x, y))
        return x, y


    def get_longest_seq_length(self, data: List[Dict]) -> Tuple[int, int]:
        # find out the minimum max_seq_length required during fine-tuning (saves memory!)
        lengths = [len(d["input_ids"]) for d in data]
        longest_seq_length = max(lengths)
        longest_seq_ix = lengths.index(longest_seq_length)
        return longest_seq_length, longest_seq_ix

