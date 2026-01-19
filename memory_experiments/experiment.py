import math
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
from typing import cast
from tqdm.auto import tqdm
import json
from datetime import datetime


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_scheduler
)
from transformers.models.llama.configuration_llama import LlamaConfig
from bitsandbytes.optim.adamw import AdamW8bit

import datasets
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from torch.utils.data import DataLoader

from memory_approaches.infini_attention import InfiniAttentionLlamaForCausalLM
from memory_approaches.infini_armt_attention import InfiniARMTLlamaForCausalLM
from generate_dataset import DatasetGenerator
from throughput_meter import ThroughputMeter


class Experiment:
    def __init__(self, config):
        self.config = config
        self.bnb_config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger(__name__)

        accelerator_log_kwargs = {}
        if self.config.with_tracking:
            accelerator_log_kwargs["log_with"] = self.config.report_to
            accelerator_log_kwargs["project_dir"] = self.config.output_dir
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            **accelerator_log_kwargs
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        if self.config.seed is not None:
            set_seed(self.config.seed)

        if self.accelerator.is_main_process and self.config.output_dir is not None:
            os.makedirs(self.config.output_dir, exist_ok=True)

        """self.dataset_generator = DatasetGenerator(
            self.config.model_name, 
            self.config.tokenizer_name
        )"""
        self.throughput = ThroughputMeter()


    def initialize_quantization(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )


    def load_model(self):
        if self.config.config_name:
            self.model_config = LlamaConfig.from_pretrained(
                self.config.config_name,
                trust_remote_code=True,
            )
        elif self.config.model_name:
            self.model_config = AutoConfig.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )

        if self.config.memory_approach == "infini_attention":
            self.model = InfiniAttentionLlamaForCausalLM.from_pretrained(
                self.config.model_name,
                config=self.model_config,
                quantization_config=self.bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
                dtype="auto"
            )
        elif self.config.memory_approach == "infini_armt_attention":
            self.model = InfiniARMTLlamaForCausalLM.from_pretrained(
                self.config.model_name,
                config=self.model_config,
                quantization_config=self.bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
                dtype="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                config=self.model_config,
                quantization_config=self.bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
                dtype="auto"
            )

        self.model.to(self.device)  # type: ignore
        if self.config.memory_approach == "infini_armt_attention":
            self.model.model.armt_memory_gate = nn.Linear(
                self.model_config.hidden_size, 
                self.model_config.hidden_size, 
                bias=True,
                device=self.device,
                dtype=torch.bfloat16)
            init.xavier_uniform_(self.model.model.armt_memory_gate.weight)
            self.model.model.armt_memory_gate.bias.data.normal_(0, 1)
        self.model.config.use_cache = False
        torch.set_default_dtype(torch.bfloat16)

        for name, param in self.model.named_parameters():
            if name.startswith("model.embed_tokens") or "layers.0" in name:
                param.requires_grad = False
        
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Number of parameters: {num_params}")
        self.logger.info(f"Number of trainable parameters: {trainable_params}")

        total_bytes = 0
        for p in self.model.parameters():
            total_bytes += p.nelement() * p.element_size()

        self.logger.info(f"Model size: {total_bytes / (1024**3):.2f} GB")


    def load_tokenizer(self):
        if self.config.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name, 
                use_fast=True,
                trust_remote_code=True
            )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.config.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > self.model_config.max_position_embeddings:
                self.logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({self.tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, self.model_config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, self.model_config.max_position_embeddings)
            else:
                if self.config.block_size > self.tokenizer.model_max_length:
                    self.logger.warning(
                        f"The block_size passed ({self.config.block_size}) is larger than the maximum length for the model "
                        f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                    )
                self.block_size = min(self.config.block_size, self.tokenizer.model_max_length)


    def load_dataset(self):
        raw_datasets = None
        if self.config.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = cast(DatasetDict, load_dataset(self.config.dataset_name))

            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = cast(Dataset, load_dataset(
                    self.config.dataset_name,
                    split=f"train[:30%]",
                ))
                raw_datasets["train"] = cast(Dataset, load_dataset(
                    self.config.dataset_name,
                    split=f"train[30%:]",
                ))

        if raw_datasets is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(1000))

        return raw_datasets


    def preprocess_dataset(self, raw_datasets):
        column_names = raw_datasets["train"].column_names if raw_datasets is not None else list()
        self.text_column_name = "text" if "text" in column_names else column_names[0]

        tokenizer = self.tokenizer
        text_column_name = self.text_column_name
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_attention_mask=False, truncation=False)
    
        block_size = self.config.block_size
        def generate_segments(examples):
            input_ids_list = []
            labels_list = []

            for i in range(len(examples["input_ids"])):
                input_ids = examples["input_ids"][i]

                total_length = len(input_ids)
                total_length = (total_length // (block_size - 10)) * (block_size - 10)
                
                chunks = [input_ids[j : j + block_size - 10] for j in range(0, total_length, block_size - 10)]
                input_ids_list.extend(chunks)
                labels_list.extend(chunks.copy())

            return {
                "input_ids": input_ids_list,
                "labels": labels_list
            }

        segment_length = self.config.segment_length
        context_length = self.config.segment_length // 2
        num_segments = block_size // segment_length
        system_prompt = "You are a helpful assistant."
        user_prompt = "Continue the following text:\n\n{}"

        def to_chat_template(example):
            all_input_ids, all_labels = [], []
            for i in range(len(example["input_ids"])):
                original_input_ids = example["input_ids"][i]

                for seg_idx in range(num_segments):
                    start = seg_idx * segment_length
                    end = start + segment_length
                    segment_input_ids = original_input_ids[start:end]

                    context_ids = segment_input_ids[: context_length]
                    target_ids = segment_input_ids[context_length :]

                    context_text = tokenizer.decode(context_ids)
                    target_text = tokenizer.decode(target_ids)

                    chat_text = (
                        f"<|system|>\n{system_prompt}\n"
                        f"<|user|>\n{user_prompt.format(context_text)}\n"
                    )
                    tokenized_chat_text = tokenizer(chat_text, return_attention_mask=False, truncation=False)
                
                    assistant_text = f"<|assistant|>\n{target_text}\n"
                    tokenized_assistant_text = tokenizer(assistant_text, return_attention_mask=False, truncation=False)
                    input_ids = tokenized_chat_text["input_ids"] + tokenized_assistant_text["input_ids"]
                    labels = [-100] * len(tokenized_chat_text["input_ids"]) + tokenized_assistant_text["input_ids"].copy()

                    if len(input_ids) > segment_length:
                        input_ids = input_ids[:segment_length]
                        labels = labels[:segment_length]

                    if len(input_ids) < segment_length:
                        pad_len = segment_length - len(input_ids)
                        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                        labels = labels + [-100] * pad_len

                    all_input_ids.append(input_ids)
                    all_labels.append(labels)

            return {
                "input_ids": all_input_ids,
                "labels": all_labels
            }
        
        def collate_fn(batch):
            input_ids = [torch.tensor(x["input_ids"]) for x in batch]
            labels = [torch.tensor(x["labels"]) for x in batch]

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

            return {"input_ids": input_ids, "labels": labels}

        with self.accelerator.main_process_first():
            tokenized_datasets = None
            if raw_datasets is not None:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    batch_size=self.config.batch_size,
                    num_proc=self.config.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=True,
                    desc="Running tokenizer on dataset",
                )
        
        with self.accelerator.main_process_first():
            lm_datasets = None
            if tokenized_datasets is not None:
                lm_datasets = tokenized_datasets.map(
                    generate_segments,
                    batched=True,
                    batch_size=self.config.batch_size,
                    num_proc=self.config.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc=f"Grouping texts in chunks of {self.config.block_size}",
                )

        with self.accelerator.main_process_first():
            chat_lm_datasets = None
            if lm_datasets is not None:
                chat_lm_datasets = lm_datasets.map(
                    to_chat_template,
                    batched=True,
                    batch_size=self.config.batch_size,
                    num_proc=self.config.preprocessing_num_workers,
                    load_from_cache_file=True,
                    desc="Turning data into chat template",
                )

        """anchor_dataset = self.dataset_generator.generate_anchor_dataset(int(len(chat_lm_datasets["train"]) * 0.05) if chat_lm_datasets is not None else 1000)
        if chat_lm_datasets is not None and anchor_dataset is not None:
            self.train_dataset = concatenate_datasets(
                [chat_lm_datasets["train"], anchor_dataset]
            ).shuffle(seed=42)"""

        self.train_dataset = chat_lm_datasets["train"] if chat_lm_datasets is not None else None
        self.eval_dataset = chat_lm_datasets["validation"] if chat_lm_datasets is not None else None
        self.test_dataset = chat_lm_datasets["test"] if chat_lm_datasets is not None else None
        if self.train_dataset is None or self.eval_dataset is None or self.test_dataset is None:
            raise ValueError("train_dataset or eval_dataset is None. Please check the dataset loading step.")
        
        self.train_dataloader = DataLoader(
            self.train_dataset,  # type: ignore
            shuffle=False, 
            collate_fn=collate_fn,
            batch_size=self.config.block_size // self.config.segment_length 
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset, 
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=self.config.block_size // self.config.segment_length 
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, 
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=self.config.block_size // self.config.segment_length
        )
        

    def general_training_setup(self):
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW8bit(optimizer_grouped_parameters, lr=self.config.learning_rate)
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        self.criterion = nn.CrossEntropyLoss()

        self.number_of_training_steps = self.config.number_of_epochs * len(self.train_dataloader)
        self.scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.max_train_steps
        )


    def save_model(self, perplexity):
        self.accelerator.wait_for_everyone()
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            self.config.output_dir + "/" + timestamp_str, 
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            safe_serialization=False
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.config.output_dir + "/" + timestamp_str)
            with open(os.path.join(self.config.output_dir, timestamp_str, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


    def resume_from_checkpoint(self, best_checkpoint=""):
        if best_checkpoint == "":
            checkpoint_path = self.config.resume_from_checkpoint
        else:
            checkpoint_path = best_checkpoint
        
        path = os.path.basename(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        self.accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if best_checkpoint == "":
            if "epoch" in training_difference:
                self.starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                self.resume_step = None
                self.completed_steps = self.starting_epoch * self.num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                self.resume_step = int(training_difference.replace("step_", "")) * self.config.gradient_accumulation_steps
                self.starting_epoch = self.resume_step // len(self.train_dataloader)
                self.completed_steps = self.resume_step // self.config.gradient_accumulation_steps
                self.resume_step -= self.starting_epoch * len(self.train_dataloader)


    def list_params_with_grads(self):
        params_with_grad = []
        params_without_grad = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                params_with_grad.append(name)
            else:
                params_without_grad.append(name)

        return params_with_grad, params_without_grad


    def train_loop(self):
        if self.config.with_tracking:
            experiment_config = dict(vars(self.config))
            self.accelerator.init_trackers(
                project_name=os.environ["WANDB_PROJECT"],
                config=experiment_config,
                init_kwargs={
                    "wandb": {
                        "entity": os.environ["WANDB_ENTITY"],
                    }}
            )
    
        total_batch_size = (self.config.block_size // self.config.segment_length) * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataloader)}")
        self.logger.info(f"  Num Epochs = {self.config.number_of_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.config.block_size // self.config.segment_length}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Total optimization steps = {self.config.max_train_steps}")

        progress_bar = tqdm(range(self.config.max_train_steps), disable=not self.accelerator.is_local_main_process)
        self.completed_steps = 0
        self.starting_epoch = 0
        best_checkpoint_path = ""
        self.model.to(self.device)  # type: ignore
        
        if self.config.resume_from_checkpoint:
            self.resume_from_checkpoint()

        progress_bar.update(self.completed_steps)
        perplexity, best_val_loss = None, float("inf")
        self.throughput.start()

        for epoch in range(self.starting_epoch, self.config.number_of_epochs):
            print("Epoch ", epoch + 1, "/", self.config.number_of_epochs)
            self.model.train()
            
            total_train_loss, loss = 0, torch.empty(0)
            for _, batch in enumerate(self.train_dataloader):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                total_block_loss = 0
                self.model.reset_memory()  # type: ignore

                for i in range(input_ids.shape[0]):
                    outputs = None
                    segment_input_ids = input_ids[i].unsqueeze(0)
                    segment_labels = labels[i].unsqueeze(0)
                    if self.completed_steps % 100 == 0:
                        torch.cuda.reset_peak_memory_stats()
                        
                    outputs = self.model(
                        input_ids=segment_input_ids, 
                        labels=segment_labels,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True)

                    if outputs is None:
                        raise ValueError("Outputs is None. Please check the model forward pass.")
                    
                    loss = outputs.loss
                    total_train_loss += loss.detach().item()
                    total_block_loss += loss.detach().item()

                    num_tokens = (labels[i] != -100).sum().item() if labels is not None else 0
                    self.throughput.update(num_tokens)

                if self.completed_steps % 50 == 0:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1e9  # no clipping, just measuring
                    )
                    metrics = {}
                    metrics["train_grad_norm"] = grad_norm.item() if grad_norm is not None else 0.0
                    self.accelerator.log(metrics, step=self.completed_steps)

                avg_segment_loss = total_block_loss / len(input_ids)
                self.accelerator.log(
                    {"average_segment_loss": avg_segment_loss},
                    step=self.completed_steps,
                )
                
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.completed_steps % 100 == 0:
                    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                    tokens_per_sec = self.throughput.compute()

                    if self.accelerator.is_main_process:
                        self.accelerator.log(
                            {"train_peak_gpu_mem_mb": peak_mem,
                             "train_tokens_per_sec": tokens_per_sec},
                            step=self.completed_steps,
                        )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                metrics = {
                    "train_loss": loss.detach().float().item(),
                }
                self.accelerator.log(metrics, step=self.completed_steps)
                if self.completed_steps % 20 == 0:
                    metrics = {
                        "train_lr": self.scheduler.get_last_lr()[0]
                    }
                    self.accelerator.log(metrics, step=self.completed_steps)

                if self.completed_steps % 500 == 0:
                    memory = self.model.get_memory()  # type: ignore
                    detached_memory = memory.detach()  # type: ignore

                    self.accelerator.log(
                        {
                            "train_memory_norm_mean": detached_memory.norm(dim=-1).mean().item(),
                            "train_memory_norm_max": detached_memory.norm(dim=-1).max().item(),
                        },
                        step=self.completed_steps
                    )

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.completed_steps += 1

                if isinstance(self.config.checkpointing_steps, int):
                    if self.completed_steps % self.config.checkpointing_steps == 0:
                        output_dir = f"epoch_{epoch}_step_{self.completed_steps}"
                        if self.config.output_dir is not None:
                            output_dir = os.path.join(self.config.output_dir, output_dir)
                        self.accelerator.save_state(output_dir)

                if self.completed_steps >= (epoch + 1) * self.config.max_train_steps / self.config.number_of_epochs:
                    break
                
            self.model.eval()
            torch.cuda.reset_peak_memory_stats()
            total_val_loss, total_tokens, self.valid_completed_steps = 0, 0, 0
            for _, batch in enumerate(self.eval_dataloader):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                
                for i in range(input_ids.shape[0]):
                    with torch.no_grad():
                        outputs = None
                        segment_input_ids = input_ids[i].unsqueeze(0)
                        segment_labels = labels[i].unsqueeze(0)

                        outputs = self.model(
                            input_ids=segment_input_ids, 
                            labels=segment_labels if labels is not None else None,
                            use_cache = False)

                    if outputs is None:
                        raise ValueError("Outputs is None. Please check the model forward pass.")
                    
                    loss = cast(torch.Tensor, outputs.loss)
                    batch_labels = labels[i] if labels is not None else None
                    num_tokens = (batch_labels != -100).sum()

                    total_val_loss += self.accelerator.gather_for_metrics(
                        loss * num_tokens
                    ).sum().item()  # type: ignore

                    total_tokens += self.accelerator.gather_for_metrics(
                        num_tokens
                    ).sum().item()  # type: ignore

                    self.throughput.update(num_tokens)
                    self.valid_completed_steps += 1
                    if self.valid_completed_steps >= self.config.max_valid_steps / self.config.number_of_epochs:
                        break
                
            val_loss = total_val_loss / total_tokens
            if val_loss < best_val_loss:
                output_dir = f"best_epoch_{epoch}"
                if self.config.output_dir is not None:
                    output_dir = os.path.join(self.config.output_dir, output_dir)
                self.accelerator.save_state(output_dir)
                best_checkpoint_path = output_dir

            try:
                perplexity = math.exp(val_loss)
            except OverflowError:
                perplexity = float("inf")

            memory = self.model.get_memory()  # type: ignore
            if memory is not None:  # type: ignore
                detached_memory = memory.detach()  # type: ignore
                memory_stats = {
                    "memory_norm_mean": detached_memory.norm(dim=-1).mean().item(),
                    "memory_norm_max": detached_memory.norm(dim=-1).max().item(),
                    "memory_sparsity": (detached_memory.abs() < 1e-4).float().mean().item(),
                }

                self.accelerator.log(
                    {f"eval_{k}": v for k, v in memory_stats.items()}
                )

            self.logger.info(f"Epoch {epoch + 1}: perplexity: {perplexity} val_loss: {val_loss}")
            eval_tokens_per_sec = self.throughput.compute()
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

            if self.config.with_tracking:
                self.accelerator.log(
                    {
                        "perplexity": perplexity,
                        "val_loss": val_loss,
                        "total_train_loss": total_train_loss / len(self.train_dataloader),
                        "eval_tokens_per_sec": eval_tokens_per_sec,
                        "eval_peak_gpu_mem_mb": peak_mem_mb
                    }
                )
            self.completed_steps += 1

        self.resume_from_checkpoint(best_checkpoint_path)
        self.model.eval()
        total_test_loss, total_tokens = 0, 0
        self.test_completed_steps = 0

        progress_bar = tqdm(range(self.config.max_test_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.update(self.test_completed_steps)
        
        for _, batch in enumerate(self.test_dataloader):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            for i in range(input_ids.shape[0]):
                with torch.no_grad():
                    outputs = None
                    segment_input_ids = input_ids[i].unsqueeze(0).to(self.device)
                    segment_labels = labels[i].unsqueeze(0).to(self.device)

                    outputs = self.model(
                        input_ids=segment_input_ids, 
                        labels=segment_labels,
                        use_cache=False)
                
                if outputs is None:
                    raise ValueError("Outputs is None. Please check the model forward pass.")
                
                loss = cast(torch.Tensor, outputs.loss)
                batch_labels = labels[i] if labels is not None else None
                num_tokens = (batch_labels != -100).sum()

                total_test_loss += self.accelerator.gather_for_metrics(
                    loss * num_tokens
                ).sum().item()  # type: ignore

                total_tokens += self.accelerator.gather_for_metrics(
                    num_tokens
                ).sum().item()  # type: ignore

                progress_bar.update(1)
                self.test_completed_steps += 1
                if self.test_completed_steps >= self.config.max_test_steps:
                    break
            
        test_loss = total_test_loss / total_tokens
        try:
            perplexity = math.exp(test_loss)
        except OverflowError:
            perplexity = float("inf")

        memory = self.model.get_memory()  # type: ignore
        if memory is not None:  # type: ignore
            detached_memory = memory.detach()  # type: ignore
            memory_stats = {
                "test_memory_norm_mean": detached_memory.norm(dim=-1).mean().item(),
                "test_memory_norm_max": detached_memory.norm(dim=-1).max().item(),
                "test_memory_sparsity": (detached_memory.abs() < 1e-4).float().mean().item(),
            }

            self.accelerator.log(
                {f"test_{k}": v for k, v in memory_stats.items()}
            )

        if self.config.with_tracking:
            self.accelerator.log(
                {
                    "test_perplexity": perplexity,
                    "test_loss": test_loss
                }
            )

        if self.config.with_tracking:
            self.accelerator.end_training()

        if self.config.output_dir is not None:
            self.save_model(perplexity)


    def run(self):
        if self.config.quantization:
            self.initialize_quantization()

        self.load_tokenizer()
        self.load_model()

        self.d_model = self.model.config.hidden_size
        self.number_of_layers = self.model.config.num_hidden_layers
        self.embedding_layer = cast(nn.Embedding, self.model.get_input_embeddings())
        self.embedding_size = self.embedding_layer.weight.shape[0]

        raw_datasets = self.load_dataset()
        self.preprocess_dataset(raw_datasets)
        self.general_training_setup()

        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.scheduler
        )
        self.train_loop()
