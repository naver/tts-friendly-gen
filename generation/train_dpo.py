# This file was derived and modified from HuggingFace's TRL (released
# under Apache License 2.0), based on the following file:
# https://github.com/huggingface/trl/blob/v1.0-release/trl/scripts/dpo.py
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

import torch
from datasets import load_dataset, DatasetDict
from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import random
import os
from dataclasses import dataclass, field
from typing import List
import json
from utils.prompts import *
import yaml
import sys
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

tqdm.pandas()

@dataclass
class DataConfig:
    data_dir: str = ""
    context: str = "human"
    response_prefix: str = "assistant"
    options: List[str] = field(default_factory=lambda: ["chosen", "rejected"])
    cv: bool = False # To be used if the dataset contains several splits for cross-validation
    use_template: bool = False # If True, the model's chat template is used
    num_cv: int = 5 # Number of cross-validation splits
    train_size: str = "" # Use "" for full training set
    eval_batch_size: int = 10

def get_respond(model, tokenizer, device, data_config):
    model.eval()
    
    def respond(messages):
        if data_config.use_template:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            ).to(device)
        else:
            inputs = tokenizer(
                messages,
                return_tensors="pt",
                padding=True
            ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1024
            )
        out = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        out = out.rstrip("</s>")
        out = out.strip(" ")
        out = out.strip('"')

        return out

    return respond

def get_dpo_generator(respond, data_config):

    def dpo_generator(test_situation):
        # Create the prompt containing the situation
        data_type_upper = data_config.data_type.upper()
        sys_text = eval(f"{data_type_upper}_GEN_SYS_TEMPLATE")
        user_text = eval(f"{data_type_upper}_GEN_ZEROSHOT_USER_TEMPLATE").format(situation=test_situation[0])
        if data_config.use_template:
            messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}]
        else:
            messages = f"{sys_text}\n\n{user_text}"

        # Generate the response containing the action for the situation
        out = respond(messages)
        return out

    return dpo_generator

def preprocess_function(examples, data_config, tokenizer):
    new_examples = {
        "chosen": [],
        "rejected": [],
    }
    for i in range(len(examples[data_config.context])):
        situation = examples[data_config.context][i]
        chosen_action = examples[f'{data_config.response_prefix}_chosen'][i]
        data_type_upper = data_config.data_type.upper()
        sys_text = eval(f"{data_type_upper}_GEN_SYS_TEMPLATE")
        user_text = eval(f"{data_type_upper}_GEN_ZEROSHOT_USER_TEMPLATE").format(situation=situation)

        if data_config.use_template:
            chosen_messages = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": chosen_action}
            ]
        else:
            chosen_messages = f"{sys_text}\n\n{user_text} {chosen_action}{tokenizer.eos_token}"

        for option in data_config.options:
            if option == "chosen":
                continue # Only consider rejected options to contrast with the chosen option
            rejected_action = examples[data_config.response_prefix + "_" + option][i]

            if data_config.use_template:
                rejected_messages = [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": rejected_action}
                ]
            else:
                rejected_messages = f"{sys_text}\n\n{user_text} {rejected_action}{tokenizer.eos_token}"

            new_examples["chosen"].append(chosen_messages)
            new_examples["rejected"].append(rejected_messages)

    return new_examples

def parse_ds(ds, context, response_prefix, options):
    situations = []
    for row in tqdm(ds, desc='parsing dataset'):
        situation = row[f"{context}"]
        actions = [row[f"{response_prefix}_{option}"] for option in options]
        random.shuffle(actions) # To avoid always having the chosen action first
        chosen = row[f"{response_prefix}_chosen"]
        situations.append([situation, actions, chosen])

    return situations

def get_model_outputs(model, tokenizer, ds, data_config, method_name):
    respond = get_respond(model, tokenizer, model.device, data_config)
    dpo_generator = get_dpo_generator(respond, data_config)

    sets_dict = []
    for set_name in ["train", "validation", "test"]: # Generations are also done on the train set for verifications
        eval_ds = ds[set_name] # Situations for which the generations will be made
        eval_situations = parse_ds(eval_ds, data_config.context, data_config.response_prefix, data_config.options)

        samples_dict = []
        for eval_situation in tqdm(eval_situations, desc='running DPO generator'):
            generated_action = dpo_generator(eval_situation)
            situation, actions, chosen_action = eval_situation[0], eval_situation[1], eval_situation[2]
            print("# Situation:\n", situation)
            print("# Chosen action:\n", chosen_action)
            print("# Generated action:\n", generated_action)
            samples_dict.append(
                {
                    "context": situation,
                    "choices": actions,
                    "preferred_choice": chosen_action,
                    "generated_choices": [
                        {
                            "generator_id": method_name,
                            "generated_choice": generated_action
                        }
                    ]
                }
            )
        sets_dict.append({"set_id": set_name, "samples": samples_dict})

    return sets_dict

def main():
    seed = 2024
    transformers.set_seed(seed)
    random.seed(seed)
    parser = TrlParser((DPOConfig, ModelConfig, DataConfig))
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        # Load YAML manually
        with open(sys.argv[2], "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict = {k.replace("-", "_"): v for k, v in config_dict.items()} # Automatically done in argparse
        training_args, model_config, data_config = parser.parse_dict(config_dict)
    else:
        training_args, model_config, data_config = parser.parse_args_and_config()
    training_args.save_strategy = "no" # Do not save model checkpoints
    data_config.data_type = data_config.data_dir.split("/")[-2]
    print(model_config)
    print(data_config)

    short_model_name = model_config.model_name_or_path.split("/")[-1]
    per_device_train_batch_size = training_args.per_device_train_batch_size
    num_train_epochs = int(training_args.num_train_epochs)
    beta = training_args.beta
    method_name = f"dpo_{short_model_name}_b{per_device_train_batch_size}_e{num_train_epochs}_beta{beta}"
    if data_config.use_template:
        method_name += "_ut"
    print(method_name)

    ################ Load the tokenizer ################

    dtype = (
        model_config.dtype
        if model_config.dtype in ["auto", None]
        else getattr(torch, model_config.dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        dtype=dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################ Train a DPO model for each dataset ################

    output_dict = []
    dataset_list = [dir for dir in os.listdir(data_config.data_dir)]
    dataset_list.sort()
    for dataset in dataset_list:
        splits_dict = []

        if data_config.cv:
            dataset_splits = [dataset + "/" + str(i) for i in range(data_config.num_cv)]
        else:
            dataset_splits = [dataset + "/"]

        for dataset_split in dataset_splits:
            split_id = dataset_split.split("/")[-1] if data_config.cv else ""
            print("Dataset: " + dataset_split)
            dataset_split_path = data_config.data_dir + "/" + dataset_split

            ################ Load the model ################

            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name_or_path, **model_kwargs
            )
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))

            peft_config = get_peft_config(model_config)
            if peft_config is None:
                ref_model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_name_or_path, **model_kwargs
                )
                ref_model.config.pad_token_id = tokenizer.pad_token_id
                ref_model.resize_token_embeddings(len(tokenizer))
            else:
                ref_model = None

            ################ Load & preprocess the dataset ################

            train_ds, val_ds, test_ds = load_dataset(dataset_split_path, split=[f'train[:{data_config.train_size}]', 'validation', 'test'])
            ds = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

            # Preprocess the dataset (pairwise transformation, formatting)
            pw_train_ds = ds["train"].map(
                lambda e: preprocess_function(e, data_config, tokenizer),
                batched=True,
                num_proc=4,
                remove_columns=ds["train"].column_names,
            )

            ################ Train the DPO model ################

            trainer = DPOTrainer(
                model,
                ref_model,
                args=training_args,
                train_dataset=pw_train_ds,
                processing_class=tokenizer,
                peft_config=peft_config,
            )
            if training_args.num_train_epochs > 0:
                trainer.train()

            ################ Generation & evaluation ################

            sets_dict = get_model_outputs(model, tokenizer, ds, data_config, method_name)
            splits_dict.append({"split_id": split_id, "sets": sets_dict})

            # Free memory
            del model 
            del ref_model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        output_dict.append({"profile": dataset, "splits": splits_dict})

    ################ Save outputs ################

    short_data_name = dataset_list[0].split("-")[0]
    save_fname = f'{short_data_name}{data_config.train_size}_{method_name}'

    out_path = 'generation/out/'
    os.makedirs(os.path.dirname(out_path + save_fname), exist_ok=True)

    out_fname = f'{save_fname}.json'
    with open(out_path + out_fname, "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, ensure_ascii=False, indent=2)
    print(f'Saved to {out_path + out_fname}')

main()
