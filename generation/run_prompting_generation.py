from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from datasets import load_dataset, DatasetDict
import argparse
import os
import random
import torch
from tqdm import tqdm
import json
from utils.common import fetch_profile_desc
from utils.prompts import *
import yaml

def get_respond(model_name, model_type, seed):
    if model_type == 'huggingface':
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )

        model_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        ).to("cuda")
        model.eval()

        def respond(messages):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            ).to("cuda")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=1024
                )
            out = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            out = out.rstrip("</s>")

            return out

        return respond
    elif model_type == 'openai':
        # OpenAI client
        client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

        # Define the text completion function
        respond = lambda messages: client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            seed=seed,
            messages=messages
        ).choices[0].message.content.strip('"')

        return respond
    else:
        return None

def get_prompting_generator(respond, data_type):

    def prompting_generator(test_situation, profile_desc):
        # Create the prompt containing the situation
        data_type_upper = data_type.upper()
        sys_text = eval(f"{data_type_upper}_GEN_SYS_TEMPLATE")
        user_text = eval(f"{data_type_upper}_GEN_ORACLEGEN_USER_TEMPLATE").format(situation=test_situation[0], profile_desc=profile_desc)
        messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}]

        # Generate the response containing the action for the new situation
        out = respond(messages)
        return out

    return prompting_generator

def parse_ds(ds, context, response_prefix, options):
    situations = []
    for row in tqdm(ds, desc='parsing dataset'):
        situation = row[f"{context}"]
        actions = [row[f"{response_prefix}_{option}"] for option in options]
        random.shuffle(actions) # To avoid always having the chosen action first
        chosen = row[f"{response_prefix}_chosen"]
        situations.append([situation, actions, chosen])

    return situations

def get_model_outputs(prompting_generator, ds, context, response_prefix, options, method_name, profile_desc):
    sets_dict = []
    for set_name in ["train", "validation", "test"]: # Generations are also done on the train set for verifications
        eval_ds = ds[set_name] # Situations for which the generations will be made
        eval_situations = parse_ds(eval_ds, context, response_prefix, options)

        samples_dict = []
        for eval_situation in tqdm(eval_situations, desc='running oracle generator'):
            generated_action = prompting_generator(eval_situation, profile_desc)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='path to YAML config file (if other arguments are not specified)')
    ap.add_argument('--model-name', default='Qwen/Qwen3-4B-Instruct-2507')
    ap.add_argument('--model-type', default='huggingface', choices=['openai', 'huggingface'])
    ap.add_argument('--data-dir', default='')
    ap.add_argument('--options', nargs='+', default=['chosen', 'rejected'])
    ap.add_argument('--context', default='human')
    ap.add_argument('--response-prefix', default='assistant')
    ap.add_argument('--cv', action='store_true', help='to be used if the dataset contains several splits for cross-validation')
    ap.add_argument('--num-cv', default=5, type=int, help='number of cross-validation splits')
    ap.add_argument('--profile-path', default='', help='')
    args = ap.parse_args()

    # Load YAML if provided
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config_dict = {k.replace("-", "_"): v for k, v in config_dict.items()} # Automatically done in argparse
        for k, v in config_dict.items():
            setattr(args, k, v)
    print(args)

    seed = 2024
    random.seed(seed)

    ################ Prepare model ################

    data_type = args.data_dir.split("/")[-2]
    short_data_name = args.data_dir.split("/")[-1] # May differ from data_type
    respond = get_respond(args.model_name, args.model_type, seed)
    prompting_generator = get_prompting_generator(respond, data_type)
    short_model_name = args.model_name.split("/")[-1]
    method_name = f"oracle_{short_model_name}"

    ################ Fetch each profile's description from the profile file ################

    profile_desc_dict = fetch_profile_desc(args.profile_path, short_data_name)

    ################ Run oracle generation for each dataset ################

    output_dict = []
    dataset_list = [dir for dir in os.listdir(args.data_dir)]
    dataset_list.sort()
    for dataset in dataset_list:
        splits_dict = []

        if args.cv:
            dataset_splits = [dataset + "/" + str(i) for i in range(args.num_cv)]
        else:
            dataset_splits = [dataset + "/"]

        # Fetch the description for the current profile
        profile_desc = profile_desc_dict[dataset]

        for dataset_split in dataset_splits:
            split_id = dataset_split.split("/")[-1] if args.cv else ""
            print("Dataset: " + dataset_split)
            dataset_split_path = args.data_dir + "/" + dataset_split

            ################ Load the dataset ################

            train_ds, val_ds, test_ds = load_dataset(dataset_split_path, split=[f'train', 'validation', 'test'])
            ds = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

            ################ Generation ################

            sets_dict = get_model_outputs(prompting_generator, ds, args.context, args.response_prefix, args.options, method_name, profile_desc)
            splits_dict.append({"split_id": split_id, "sets": sets_dict})

        output_dict.append({"profile": dataset, "splits": splits_dict})

    ################ Save outputs ################

    save_fname = f'{short_data_name}_{method_name}'

    out_path = 'generation/out/'
    os.makedirs(os.path.dirname(out_path + save_fname), exist_ok=True)

    out_fname = f'{save_fname}.json'
    with open(out_path + out_fname, "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, ensure_ascii=False, indent=2)
    print(f'Saved to {out_path + out_fname}')

main()
