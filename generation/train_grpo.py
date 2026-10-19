# This file was derived and modified from HuggingFace's TRL (released
# under Apache License 2.0), based on the following file:
# https://github.com/huggingface/trl/blob/v1.0-release/trl/scripts/grpo.py
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
import time
import yaml
import sys
from utils.reward_model import FeatureAwareRewardModel, RewardModel
from utils.prompts import *
from trl import (
    ModelConfig,
    GRPOConfig,
    GRPOTrainer,
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
    cv: bool = False
    use_template: bool = False
    num_cv: int = 5
    train_size: str = ""
    eval_batch_size: int = 10
    eval_temperature: float = 0.7
    eval_top_k: int = 20
    eval_top_p: float = 0.8
    rm_type: str = "farm"
    rm_path: str = ""
    rm_name: str = ""


def get_respond(model, tokenizer, device, data_config, temperature, top_k, top_p):
    """Returns a callable that generates responses from the model."""
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
            out = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            out = [sample.rstrip("</s>").strip(" ") for sample in out]

        return out

    return respond


def get_grpo_generator(respond, data_config):
    """Returns a callable that generates responses for a given situation."""
    def grpo_generator(test_situation, num_samples=1):
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

        return respond(messages, batch_inference=(num_samples > 1))

    return grpo_generator


def parse_ds(ds, context, response_prefix, options):
    situations = []
    for row in tqdm(ds, desc='parsing dataset'):
        situation = row[f"{context}"]
        actions = [row[f"{response_prefix}_{option}"] for option in options]
        random.shuffle(actions) # To avoid always having the chosen action first
        chosen = row[f"{response_prefix}_chosen"]
        situations.append([situation, actions, chosen])
    return situations


def make_grpo_dataset(raw_ds, data_config):
    """
    Convert a HuggingFace dataset split into the format expected by GRPOTrainer.

    GRPOTrainer expects each example to have a "prompt" key containing a list of
    chat messages (or a plain string if not using a template).  The trainer will
    generate `num_generations` completions per prompt and pass them together with
    the prompts to the reward function.
    """
    data_type_upper = data_config.data_type.upper()
    sys_text = eval(f"{data_type_upper}_GEN_SYS_TEMPLATE")

    def build_prompt(example):
        situation = example[data_config.context]
        user_text = eval(f"{data_type_upper}_GEN_ZEROSHOT_USER_TEMPLATE").format(situation=situation)

        if data_config.use_template:
            prompt = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user_text},
            ]
        else:
            prompt = f"{sys_text}\n\n{user_text}"

        return {"prompt": prompt, "situation": situation}

    return raw_ds.map(build_prompt, remove_columns=raw_ds.column_names)


def build_reward_fn(reward_model):
    """
    Wrap the reward model into the signature expected by GRPOTrainer:

        reward_fn(prompts, completions, **kwargs) -> List[float]

    GRPOTrainer calls this once per batch.  `prompts` and `completions` are
    both lists of strings (or lists of message dicts when use_template=True,
    but by that point the trainer has already decoded them to strings).
    The "situation" column is forwarded as a keyword argument because we keep
    it in the dataset (see make_grpo_dataset).
    """
    def reward_fn(prompts, completions, situation, **kwargs):
        scores = []
        for i, completion_msg in enumerate(completions):
            ctx = situation[i]
            assert completion_msg[0]['role'] == 'assistant'
            completion = completion_msg[0]['content']
            score = reward_model.score(ctx, completion)
            scores.append(float(score))
        return scores

    return reward_fn


def get_model_outputs(model, tokenizer, ds, data_config, method_name):
    respond = get_respond(
        model, tokenizer, model.device, data_config,
        temperature=data_config.eval_temperature,
        top_k=data_config.eval_top_k,
        top_p=data_config.eval_top_p
    )
    grpo_generator = get_grpo_generator(respond, data_config)
    sets_dict = []
    for set_name in ["train", "validation", "test"]:
        eval_ds = ds[set_name]
        eval_situations = parse_ds(eval_ds, data_config.context, data_config.response_prefix, data_config.options)

        samples_dict = []
        for eval_situation in tqdm(eval_situations, desc='running GRPO generator'):
            generated_action = grpo_generator(eval_situation)
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

    # ── Parse CLI / YAML config ──────────────────────────────────────────────
    parser = TrlParser((GRPOConfig, ModelConfig, DataConfig))
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        with open(sys.argv[2], "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict = {k.replace("-", "_"): v for k, v in config_dict.items()}
        training_args, model_config, data_config = parser.parse_dict(config_dict)
    else:
        training_args, model_config, data_config = parser.parse_args_and_config()

    training_args.save_strategy = "no"
    training_args.learning_rate = float(training_args.learning_rate)
    device = training_args.device
    data_config.data_type = data_config.data_dir.split("/")[-2]
    peft_config = get_peft_config(model_config)
    print(model_config)
    print(data_config)

    # ── Build run / file name ────────────────────────────────────────────────
    short_model_name = model_config.model_name_or_path.split("/")[-1]
    per_device_train_batch_size = training_args.per_device_train_batch_size
    num_train_epochs = training_args.num_train_epochs
    num_iterations = training_args.num_iterations
    learning_rate = float(training_args.learning_rate)
    num_generations = training_args.num_generations
    temperature = training_args.temperature
    top_k = training_args.top_k
    top_p = training_args.top_p
    beta = training_args.beta
    rm_type = data_config.rm_type.lower()
    rm_name = data_config.rm_name
    lora_params = ""
    if model_config.use_peft:
        if short_model_name.startswith("Phi-4"):
            peft_config.target_modules = ["qkv_proj"]
        lora_r = model_config.lora_r
        lora_alpha = model_config.lora_alpha
        lora_params += f"_lora-r{lora_r}-alpha{lora_alpha}"
    method_name = (
        f"grpo-{rm_type}_{rm_name}_{short_model_name}{lora_params}"
        f"_b{per_device_train_batch_size}_e{num_train_epochs}_i{num_iterations}_l{learning_rate}_s{num_generations}_t{temperature}_k{top_k}_p{top_p}_beta{beta}"
    )
    print(method_name)

    # ── Tokenizer ────────────────────────────────────────────────────────────
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
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True
    )
    add_pad_token = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        add_pad_token = True

    data_type_upper = data_config.data_type.upper()

    # ── Main training loop ───────────────────────────────────────────────────
    overall_start_time = time.time()
    output_dict = []
    dataset_list = sorted(os.listdir(data_config.data_dir))

    for dataset in dataset_list:
        splits_dict = []

        dataset_splits = (
            [dataset + "/" + str(i) for i in range(data_config.num_cv)]
            if data_config.cv
            else [dataset + "/"]
        )

        for dataset_split in dataset_splits:
            split_id = dataset_split.split("/")[-1] if data_config.cv else ""
            print("Dataset: " + dataset_split)
            dataset_split_path = data_config.data_dir + "/" + dataset_split

            # ── Load model ───────────────────────────────────────────────────
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name_or_path, **model_kwargs
            ).to(device)
            if add_pad_token:
                model.config.pad_token_id = tokenizer.pad_token_id
                model.resize_token_embeddings(len(tokenizer))

            # ── Load reward model ────────────────────────────────────────────
            if rm_type == "farm":
                rm_rel_path = "/" + dataset + "/data/" + split_id + "/save.pkl"
                reward_model = FeatureAwareRewardModel(
                    data_config.rm_path + rm_rel_path, torch.bfloat16
                )
            elif rm_type == "rm":
                rm_rel_path = "/" + dataset + "/" + split_id + "/reward.pt"
                reward_model = RewardModel(
                    data_config.rm_path + rm_rel_path, data_type_upper, use_template=True
                )
            else:
                print("Invalid reward model type")
                exit()

            # ── Load dataset ─────────────────────────────────────────────────
            train_ds, val_ds, test_ds = load_dataset(
                dataset_split_path,
                split=[f'train[:{data_config.train_size}]', 'validation', 'test']
            )
            ds = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

            # ── Build GRPO prompt dataset ─────────────────────────────────────
            grpo_train_ds = make_grpo_dataset(train_ds, data_config)
            reward_fn = build_reward_fn(reward_model)

            # ── GRPO training ─────────────────────────────────────────────────
            trainer = GRPOTrainer(
                model=model,
                args=training_args,
                train_dataset=grpo_train_ds,
                processing_class=tokenizer,
                reward_funcs=reward_fn,
                peft_config=peft_config,
            )
            trainer.train()

            # ── Evaluation / generation ───────────────────────────────────────
            sets_dict_out = get_model_outputs(model, tokenizer, ds, data_config, method_name)
            splits_dict.append({"split_id": split_id, "sets": sets_dict_out})

            # ── Free memory ───────────────────────────────────────────────────
            del model
            del reward_model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        output_dict.append({"profile": dataset, "splits": splits_dict})

    overall_end_time = time.time()
    total_runtime = overall_end_time - overall_start_time
    print(f"Total training runtime: {total_runtime:.2f} seconds")
    print("Max allocated memory (MB):", torch.cuda.max_memory_allocated() / 1024**2)

    # ── Save outputs ──────────────────────────────────────────────────────────
    short_data_name = dataset_list[0].split("-")[0]
    save_fname = f'{short_data_name}{data_config.train_size}_{method_name}'
    out_path = 'generation/out/'
    os.makedirs(os.path.dirname(out_path + save_fname), exist_ok=True)

    out_fname = f'{save_fname}.json'
    with open(out_path + out_fname, "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, ensure_ascii=False, indent=2)
    print(f'Saved to {out_path + out_fname}')


main()
