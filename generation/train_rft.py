# This file was derived and modified from HuggingFace's TRL (released
# under Apache License 2.0), based on the following file:
# https://github.com/huggingface/trl/blob/v1.0-release/trl/scripts/sft.py
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
from copy import deepcopy
from utils.reward_model import FeatureAwareRewardModel, RewardModel
from utils.prompts import *
import time
import yaml
import sys
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map
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
    num_train_iters: int = 5 # Number of iterations in RFT
    num_samples: int = 10 # Samples generated in the data generation phase
    train_temperature: float = 1.2 # Temperature used in the data generation phase
    train_top_k: int = 0 # Top-k used in the data generation phase
    train_top_p: float = 0.9 # Top-p used in the data generation phase
    eval_temperature: float = 0.7 # Temperature used in the evaluation
    eval_top_k: int = 20 # Top-k used in the evaluation
    eval_top_p: float = 0.8 # Top-p used in the evaluation
    add_prev_chosen: bool = False # If True, the chosen response from the previous round is added to the sample set to select from
    num_train_epochs_per_iter: int = 5 # Number of iterations in RFT
    rm_type: str = "farm" # Type of reward model used (either "farm" for feature-aware reward model or "rm" for traditional reward model)
    rm_path: str = "" # Path of the directory containing the reward model
    rm_name: str = "" # Name of the reward model (only for logging purpose)

def get_respond(model, tokenizer, device, data_config, temperature, top_k, top_p):
    model.eval()
    
    def respond(messages, batch_inference=False):
        if data_config.use_template:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
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
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=1024
            )
        if not batch_inference:
            out = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            out = out.rstrip("</s>").strip(" ")
        else:
            # The prefix is assumed to be shared for all batch samples
            out = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            out = [sample.rstrip("</s>").strip(" ") for sample in out]

        return out

    return respond

def get_rft_generator(respond, data_config):

    def rft_generator(test_situation, num_samples=1):
        # Create the prompt containing the situation
        data_type_upper = data_config.data_type.upper()
        sys_text = eval(f"{data_type_upper}_GEN_SYS_TEMPLATE")
        user_text = eval(f"{data_type_upper}_GEN_ZEROSHOT_USER_TEMPLATE").format(situation=test_situation[0])
        if data_config.use_template:
            if num_samples > 1:
                messages = [[{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}] for _ in range(num_samples)]
            else:
                messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}]
        else:
            if num_samples > 1:
                messages = [f"{sys_text}\n\n{user_text}" for _ in range(num_samples)]
            else:
                messages = f"{sys_text}\n\n{user_text}"

        # Generate the response containing the action for the situation
        if num_samples > 1:
            out = respond(messages, batch_inference=True)
        else:
            out = respond(messages, batch_inference=False)
        return out

    return rft_generator

def formatting_prompts_func(example, data_config, tokenizer):
    situation = example[data_config.context]
    chosen_action = example[f'{data_config.response_prefix}_chosen']

    data_type_upper = data_config.data_type.upper()
    sys_text = eval(f"{data_type_upper}_GEN_SYS_TEMPLATE")
    user_text = eval(f"{data_type_upper}_GEN_ZEROSHOT_USER_TEMPLATE").format(
        situation=situation
    )

    if data_config.use_template:
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": chosen_action}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False
        )
    else:
        text = f"{sys_text}\n\n{user_text} {chosen_action}{tokenizer.eos_token}"

    return text

def parse_ds(ds, context, response_prefix, options):
    situations = []
    for row in tqdm(ds, desc='parsing dataset'):
        situation = row[f"{context}"]
        actions = [row[f"{response_prefix}_{option}"] for option in options]
        random.shuffle(actions) # To avoid always having the chosen action first
        chosen = row[f"{response_prefix}_chosen"]
        situations.append([situation, actions, chosen])

    return situations

def data_generation(iter_train_ds, model, tokenizer, data_config):
    respond = get_respond(model, tokenizer, model.device, data_config, temperature=data_config.train_temperature, top_k=data_config.train_top_k, top_p=data_config.train_top_p)
    rft_generator = get_rft_generator(respond, data_config)
    train_situations = parse_ds(iter_train_ds, data_config.context, data_config.response_prefix, data_config.options)

    samples = []
    for train_situation in tqdm(train_situations, desc='data generation'):
        generated_actions = rft_generator(train_situation, data_config.num_samples) # Do batch inference to collect all samples at once
        situation, actions, chosen_action = train_situation[0], train_situation[1], train_situation[2]
        if data_config.add_prev_chosen: # Add previous chosen action to prevent selecting a sample with a lower reward than previous iter
            generated_actions.append(chosen_action)
        print("# Situation:\n", situation)
        print("# Chosen action:\n", chosen_action)
        print("# Generated actions:\n", generated_actions)
        samples.append(generated_actions)

    return samples

def sample_selection(ds, samples, reward_model, data_config):
    # For each situation, identify the sampled action that maximizes the reward_model
    context = data_config.context
    selected_samples = []
    total_reward = 0.0
    for situation, sample_set in tqdm(zip(ds[context], samples), desc='sample selection', total=len(ds[context])):
        best_sample, best_reward = max(
            ((action, reward_model.score(situation, action)) for action in sample_set),
            key=lambda e: e[1]
        )
        print("# Situation:\n", situation)
        print("# Best sample:\n", best_sample)
        selected_samples.append(best_sample)
        total_reward += best_reward
    average_reward = total_reward / len(samples)
    print("Average reward:", average_reward)

    # Update the dataset with the new "chosen" actions
    response_prefix = data_config.response_prefix
    ds = ds.map(lambda e, idx: {f"{response_prefix}_chosen": selected_samples[idx]}, with_indices=True)
    
    return average_reward, ds

def get_model_outputs(model, tokenizer, ds, data_config, method_name):
    respond = get_respond(model, tokenizer, model.device, data_config, temperature=data_config.eval_temperature, top_k=data_config.eval_top_k, top_p=data_config.eval_top_p)
    rft_generator = get_rft_generator(respond, data_config)
    sets_dict = []
    for set_name in ["train", "validation", "test"]: # Generations are also done on the train set for verifications
        eval_ds = ds[set_name] # Situations for which the generations will be made
        eval_situations = parse_ds(eval_ds, data_config.context, data_config.response_prefix, data_config.options)

        samples_dict = []
        for eval_situation in tqdm(eval_situations, desc='running RFT generator'):
            generated_action = rft_generator(eval_situation)
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
    parser = TrlParser((SFTConfig, ModelConfig, DataConfig))
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        # Load YAML manually
        with open(sys.argv[2], "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict = {k.replace("-", "_"): v for k, v in config_dict.items()} # Automatically done in argparse
        training_args, model_config, data_config = parser.parse_dict(config_dict)
    else:
        training_args, model_config, data_config = parser.parse_args_and_config()
    training_args.save_strategy = "no" # Do not save model checkpoints
    training_args.learning_rate = float(training_args.learning_rate)
    device = training_args.device
    data_config.data_type = data_config.data_dir.split("/")[-2]
    peft_config = get_peft_config(model_config)
    print(model_config)
    print(data_config)

    short_model_name = model_config.model_name_or_path.split("/")[-1]
    per_device_train_batch_size = training_args.per_device_train_batch_size
    num_train_epochs_per_iter = int(data_config.num_train_epochs_per_iter)
    num_train_iters = int(data_config.num_train_iters)
    learning_rate = float(training_args.learning_rate)
    num_samples = int(data_config.num_samples)
    train_temperature = float(data_config.train_temperature)
    train_top_p = float(data_config.train_top_p)
    train_top_k = int(data_config.train_top_k)
    rm_type = data_config.rm_type.lower()
    rm_name = data_config.rm_name
    lora_params = ""
    if model_config.use_peft:
        if short_model_name.startswith("Phi-4"):
            peft_config.target_modules = ["qkv_proj"]
        lora_r = model_config.lora_r
        lora_alpha = model_config.lora_alpha
        lora_params += f"_lora-r{lora_r}-alpha{lora_alpha}"
    method_name = f"rft-{rm_type}_{rm_name}_{short_model_name}{lora_params}_b{per_device_train_batch_size}_e{num_train_epochs_per_iter}_i{num_train_iters}_l{learning_rate}_s{num_samples}_t{train_temperature}_k{train_top_k}_p{train_top_p}"
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
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        add_pad_token = True
    else:
        add_pad_token = False
    
    data_type_upper = data_config.data_type.upper()

    ################ Train an RFT model for each dataset ################

    overall_start_time = time.time()

    average_rewards = dict()
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
            ).to(device)
            if add_pad_token:
                model.config.pad_token_id = tokenizer.pad_token_id
                model.resize_token_embeddings(len(tokenizer))

            ################ Load the pretrained reward model ################

            if rm_type == "farm":
                rm_rel_path = "/" + dataset + "/data/" + split_id + "/save.pkl"
                dtype = torch.bfloat16
                reward_model = FeatureAwareRewardModel(data_config.rm_path + rm_rel_path, dtype)
            elif rm_type == "rm":
                rm_rel_path = "/" + dataset + "/" + split_id + "/reward.pt"
                reward_model = RewardModel(data_config.rm_path + rm_rel_path, data_type_upper, use_template=True)
            else:
                print("Invalid reward model type")
                exit()

            ################ Load the dataset ################

            train_ds, val_ds, test_ds = load_dataset(dataset_split_path, split=[f'train[:{data_config.train_size}]', 'validation', 'test'])
            ds = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

            ################ Train the RFT model ################

            iter_train_ds = deepcopy(train_ds)
            iter_lr = training_args.learning_rate # Initial learning rate for the iteration
            for _ in range(data_config.num_train_iters):
                
                ################ Rejection sampling phase ################

                # Data generation
                # input: iter_train_ds, model, tokenizer
                # output: samples = [[action_1, ... action_n], [action_1, ... action_n], ...] with n actions generated per train situation
                samples = data_generation(iter_train_ds, model, tokenizer, data_config)

                # Sample selection
                # input: iter_train_ds, samples, reward_model
                # output:
                # - average_reward obtained for the best samples selected for training situations
                # - iter_train_ds updated with best-of-n samples according to reward_model
                average_reward, iter_train_ds = sample_selection(iter_train_ds, samples, reward_model, data_config)

                # Format dataset before fine-tuning
                formatted_iter_train_ds = iter_train_ds.map(
                    lambda e: {"text": formatting_prompts_func(e, data_config, tokenizer)},
                    remove_columns=iter_train_ds.column_names
                )

                ################ Fine-tuning phase ################

                iter_training_args = deepcopy(training_args)
                iter_training_args.learning_rate = iter_lr
                iter_training_args.num_train_epochs = num_train_epochs_per_iter
                trainer = SFTTrainer(
                    model=model,
                    args=iter_training_args,
                    train_dataset=formatted_iter_train_ds,
                    processing_class=tokenizer,
                    peft_config=peft_config,
                    #formatting_func=lambda e: formatting_prompts_func(e, data_config, tokenizer)
                )
                if iter_training_args.num_train_epochs > 0:
                    trainer.train()

            average_rewards[dataset_split] = average_reward # Record the average reward in the last iteration

            ################ Generation & evaluation ################

            sets_dict = get_model_outputs(model, tokenizer, ds, data_config, method_name)
            splits_dict.append({"split_id": split_id, "sets": sets_dict})

            # Free memory
            del model 
            del reward_model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        output_dict.append({"profile": dataset, "splits": splits_dict})

    overall_end_time = time.time()
    total_runtime = overall_end_time - overall_start_time
    print(f"Total training runtime: {total_runtime:.2f} seconds")

    print("Max allocated memory (MB):", torch.cuda.max_memory_allocated() / 1024**2)

    ################ Save outputs ################

    short_data_name = dataset_list[0].split("-")[0]
    save_fname = f'{short_data_name}{data_config.train_size}_{method_name}'

    out_path = 'generation/out/'
    os.makedirs(os.path.dirname(out_path + save_fname), exist_ok=True)

    out_fname = f'{save_fname}.json'
    with open(out_path + out_fname, "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, ensure_ascii=False, indent=2)
    print(f'Saved to {out_path + out_fname}')

    print("Per-dataset average rewards:", average_rewards)
    print("Average reward over all datasets:", sum(average_rewards.values()) / len(average_rewards.values()))

main()
