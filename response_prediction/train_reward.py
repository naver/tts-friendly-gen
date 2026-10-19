# This file was derived and modified from HuggingFace's TRL (released
# under Apache License 2.0), based on the following file:
# https://github.com/huggingface/trl/blob/v1.0-release/examples/scripts/reward_modeling.py
# Its copyright and license notice are provided below.

# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import torch
import torch.profiler
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
import transformers
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List
import csv
from utils.prompts import *
import time
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import yaml

tqdm.pandas()


@dataclass
class DataConfig:
    data_dir: str = ""
    context: str = "human"
    response_prefix: str = "assistant"
    options: List[str] = field(default_factory=lambda: ["chosen", "rejected"])
    cv: bool = False # To be used if the dataset contains several splits for cross-validation
    num_cv: int = 5 # Number of cross-validation splits
    train_size: str = "" # Use "" for full training set
    eval_batch_size: int = 10

def build_conversations(ds_split, data_config):
    conversations = []
    option_map = []

    for i in range(len(ds_split[data_config.context])):
        question = ds_split[data_config.context][i]

        for option in data_config.options:
            response = ds_split[f"{data_config.response_prefix}_{option}"][i]

            conv = [
                {"role": "system", "content": eval(f"{data_config.data_type.upper()}_GEN_SYS_TEMPLATE")},
                {"role": "user", "content": eval(f"{data_config.data_type.upper()}_GEN_ZEROSHOT_USER_TEMPLATE").format(situation=question)},
                {"role": "assistant", "content": response},
            ]

            conversations.append(conv)
            option_map.append(option)

    return conversations, option_map

def evaluate_split(model, tokenizer, ds_split, data_config, device):
    model.eval()

    conversations, option_map = build_conversations(ds_split, data_config)

    # Batch tokenize
    inputs = tokenizer.apply_chat_template(
        conversations,
        return_dict=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        enable_thinking=False
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits.view(-1).cpu().numpy()

    # Rebuild scores
    scores = {option: [] for option in data_config.options}
    for score, option in zip(logits, option_map):
        scores[option].append(score)

    preferences = ["chosen"] * len(ds_split[data_config.context])

    return calc_acc(scores, preferences, data_config.options)

def preprocess_function(examples, data_config):
    new_examples = {"chosen": [], "rejected": []}
    for i in range(len(examples[data_config.context])):
        question = examples[data_config.context][i]
        
        # Chosen option
        chosen_msg = [
            {"role": "system", "content": eval(f"{data_config.data_type.upper()}_GEN_SYS_TEMPLATE")},
            {"role": "user", "content": eval(f"{data_config.data_type.upper()}_GEN_ZEROSHOT_USER_TEMPLATE").format(situation=question)},
            {"role": "assistant", "content": examples[data_config.response_prefix + "_chosen"][i]}
        ]
        
        # Rejected options
        for option in data_config.options:
            if option == "chosen": continue
            
            rejected_msg = [
                {"role": "system", "content": eval(f"{data_config.data_type.upper()}_GEN_SYS_TEMPLATE")},
                {"role": "user", "content": eval(f"{data_config.data_type.upper()}_GEN_ZEROSHOT_USER_TEMPLATE").format(situation=question)},
                {"role": "assistant", "content": examples[data_config.response_prefix + "_" + option][i]}
            ]
            
            new_examples["chosen"].append(chosen_msg)
            new_examples["rejected"].append(rejected_msg)
            
    return new_examples

def batchify(list, batch_size):
    for i in range(0, len(list), batch_size):
        yield list[i:i + batch_size]

def calc_acc(scores, preferences, options):
    hits = []
    for i in range(len(preferences)):
        chosen_score = scores[preferences[i]][i]
        rejected_scores = [scores[option][i] for option in options if option != preferences[i]]
        max_rejected_score = max(rejected_scores)

        if chosen_score > max_rejected_score: # Success
            hit = 1
        else: # Failure
            hit = 0

        hits.append(hit)

    return np.mean(hits)

if __name__ == "__main__":
    seed = 2024
    transformers.set_seed(seed)
    parser = HfArgumentParser((RewardConfig, ModelConfig, DataConfig))
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        # Load YAML manually
        with open(sys.argv[2], "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict = {k.replace("-", "_"): v for k, v in config_dict.items()} # Automatically done in argparse
        config, model_config, data_config = parser.parse_dict(config_dict)
    else:
        config, model_config, data_config = parser.parse_args_into_dataclasses()
    config.save_strategy = "no" # Do not save model checkpoints
    config._n_gpu = 1 # Disable multi-gpu in reward trainer
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    data_config.data_type = data_config.data_dir.split("/")[-2]
    print(model_config)
    print(data_config)

    peft_config = get_peft_config(model_config)

    # Create the directory that will contain the output
    short_model_name = model_config.model_name_or_path.split("/")[-1]
    per_device_train_batch_size = config.per_device_train_batch_size
    num_train_epochs = int(config.num_train_epochs)
    short_data_name = data_config.data_dir.split("/")[-1]
    prefix = f"{short_data_name}{data_config.train_size}"
    suffix = f"_{short_model_name}_b{per_device_train_batch_size}_e{num_train_epochs}"
    if model_config.use_peft:
        if short_model_name.startswith("Phi-4"):
            peft_config.target_modules = ["qkv_proj"]
        lora_r = model_config.lora_r
        lora_alpha = model_config.lora_alpha
        suffix += f"_lora_r{lora_r}_alpha{lora_alpha}"
    out_path = f'response_prediction/out/{prefix}_rm{suffix}/'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ################ Train a RM for each dataset ################

    overall_start_time = time.time()

    results_dict = dict()
    dataset_list = [dir for dir in os.listdir(data_config.data_dir)]
    print('dataset list', dataset_list)
    dataset_list.sort()
    for dataset in dataset_list:
        print('current dataset:', dataset)
        results_dict[dataset] = {"train": dict(), "validation": dict(), "test": dict()}

        if data_config.cv:
            dataset_splits = [dataset + "/" + str(i) for i in range(data_config.num_cv)]
        else:
            dataset_splits = [dataset + "/"]

        inference_dict = dict() # Will contain the evaluation results for each dataset
        for dataset_split in dataset_splits:
            print("Training on " + dataset_split)
            inference_dict[dataset_split] = dict()
            dataset_split_path = data_config.data_dir + "/" + dataset_split

            ################ Model & Tokenizer ################

            dtype = (
                model_config.dtype
                if model_config.dtype in ["auto", None]
                else getattr(torch, model_config.dtype)
            )
            quantization_config = get_quantization_config(model_config)
            model_kwargs = dict(
                revision=model_config.model_revision,
                trust_remote_code=model_config.trust_remote_code,
                dtype=dtype,
                device_map=get_kbit_device_map() if quantization_config is not None else "auto",
                quantization_config=quantization_config,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_config.model_name_or_path, num_labels=1, **model_kwargs
            )
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.config.pad_token_id = tokenizer.pad_token_id
                model.resize_token_embeddings(len(tokenizer))

            if model_config.lora_task_type != "SEQ_CLS":
                warnings.warn(
                    "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
                    " Make sure to pass --lora_task_type SEQ_CLS when using this script."
                )

            ################ Processing the dataset ################

            # Load the dataset
            train_ds, validation_ds, test_ds = load_dataset(dataset_split_path, split=[f'train[:{data_config.train_size}]', 'validation', 'test'])
            ds = DatasetDict({'train': train_ds, 'validation': validation_ds, 'test': test_ds})

            train_dataset = ds["train"].map(lambda e: preprocess_function(e, data_config), batched=True)
            validation_dataset = ds["validation"].map(lambda e: preprocess_function(e, data_config), batched=True)

            ################ Training ################

            trainer = RewardTrainer(
                model=model,
                processing_class=tokenizer,
                args=config,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                peft_config=peft_config,
            )
            device = trainer.args.device
            trainer.train()
            try:
                trainer.save_model(out_path + dataset_split + "/reward.pt")
            except Exception as err:
                print(err, file=sys.stderr)

            ################ Evaluation ################

            model.eval()
            splits = ["train", "validation", "test"]
            with torch.no_grad():
                for split in splits:
                    acc = evaluate_split(model, tokenizer, ds[split], data_config, device)
                    inference_dict[dataset_split][split + "_accuracy"] = acc

            print(dataset_split, inference_dict[dataset_split])

            # Free memory
            del model 
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        for split in ["train", "validation", "test"]:
            results_dict[dataset][split]["acc"] = 1 / data_config.num_cv * sum([inference_dict[dataset_split][f'{split}_accuracy'] for dataset_split in dataset_splits])

    overall_end_time = time.time()
    total_runtime = overall_end_time - overall_start_time
    print(f"Total training runtime: {total_runtime:.2f} seconds")

    ################ Save cross-validation results as a CSV file ################

    csv_path = out_path + "results.csv"
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")

        # Write the header row
        writer.writerow(["Dataset", "Train Acc", "Validation Acc", "Test Acc"])

        # Write the data rows
        for dataset in dataset_list:
            train_acc = results_dict[dataset]["train"]["acc"]
            validation_acc = results_dict[dataset]["validation"]["acc"]
            test_acc = results_dict[dataset]["test"]["acc"]

            writer.writerow(
                [dataset, round(train_acc, 3), round(validation_acc, 3), round(test_acc, 3)]
            )
    print(f'Results saved to {csv_path}')

