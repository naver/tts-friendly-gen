from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from datasets import load_dataset
import argparse
import os
import random
import torch
from tqdm import tqdm
import yaml

FEATURE_GEN_SYS_TEMPLATE = "You are an assistant whose job is to propose a list of features that appropriately characterize the different choices proposed in contexts taken from a preference dataset."

FEATURE_GEN_USER_TEMPLATE = (
"""You will be given a list of contexts and a list of possible choices for each of them. Your job is to suggest a set of global features that would characterize the specificities of every choice. A feature corresponds to a criterion with a score from 1 to 5 reflecting the intensity of this feature in the given choice.

Here is the list of contexts and their associated choices:

{situations}

Provide a set of {num_features} unique and diverse features (with no duplicate) for this list of contexts and choices, formatted in JSON following the example of the template below. Make sure that features are defined to be global and not specific to a particular subset of choices. They should also be related to the choices selected, not to a specific person selecting these choices.

```json
"FEATURES": {{
    "<feature name>": {{
        "attribute_desc": "<a question which describes the feature and whose answer should assess the intensity of the feature in a given choice>",
        "attr_min": "<a lowercase participle phrase describing what is expected for the lowest intensity of the feature>",
        "attr_max": "<a lowercase participle phrase describing what is expected for the highest intensity of the feature>"
    }},
    ...
}}
```
"""
)

def get_respond(model_name, model_type, seed):
    if model_type == 'huggingface':
        # Load the model and its tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).to("cuda")
        model.eval()

        # Define the text completion function
        def respond(messages):
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            ).to("cuda")
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=4096
                )
            out = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            out = out.rstrip("</s>")

            return out

        return respond
    elif model_type == 'openai':
        # Define the OpenAI client
        client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

        # Define the parameters
        args = {
            "model": model_name,
            "seed": seed
        }
        if not model_name.startswith("o"): # Not an o1 or o3 model, for which temperature is not a supported arg
            args["temperature"] = 0.0

        # Define the text completion function
        respond = lambda messages: client.chat.completions.create(messages=messages, **args).choices[0].message.content

        return respond
    else:
        return None

def get_features_generator(respond, num_features):

    def features_generator(train_situations):
        # Create a string containing the list of situations and associated actions
        situations = ""
        for (situation, actions) in train_situations:
            situations += f"# Context:\n{situation}\n" # Here we use Context/Choice to remain generic
            for (i, action) in enumerate(actions):
                situations += f"# Choice {i+1}:\n{action}\n"
            situations += "\n"

        # Create the prompt using the list of situations
        sys_text = FEATURE_GEN_SYS_TEMPLATE
        user_text = FEATURE_GEN_USER_TEMPLATE.format(situations=situations, num_features=num_features)
        print(user_text)
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text}
        ]

        # Generate the response containing the list of features
        out = respond(messages)

        return out

    return features_generator

def parse_ds(ds, context, response_prefix, options):
    situations = []
    for row in tqdm(ds, desc='parsing dataset'):
        situation = row[f"{context}"]
        actions = [row[f"{response_prefix}_{option}"] for option in options]
        random.shuffle(actions) # To avoid always having the chosen action first
        situations.append([situation, actions])

    return situations

def load_any_dataset(dataset_arg, train_size, seed):
    if dataset_arg.endswith(".json") or dataset_arg.endswith(".jsonl"):
        ds = load_dataset("json", data_files=dataset_arg, split=f"train[:{train_size}]")
    else:
        ds = load_dataset(dataset_arg, split=f"train[:{train_size}]")
    ds = ds.shuffle(seed=seed)

    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='path to YAML config file (if other arguments are not specified)')
    ap.add_argument('--model-name', default='gpt-5.4-2026-03-05')
    ap.add_argument('--model-type', default='openai', choices=['openai', 'huggingface'])
    ap.add_argument('--dataset', default='')
    ap.add_argument('--options', nargs='+', default=['chosen', 'rejected'])
    ap.add_argument('--context', default='human')
    ap.add_argument('--response-prefix', default='assistant')
    ap.add_argument('--train-size', default="", type=str, help='number of training samples (use "" for all samples)')
    ap.add_argument('--num-features', default=40, type=int, help='number of features to generate')
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

    respond = get_respond(args.model_name, args.model_type, seed)
    features_generator = get_features_generator(respond, args.num_features)

    ################ Load the dataset ################

    train_ds = load_any_dataset(args.dataset, args.train_size, seed)

    ################ Generate features ################

    train_situations = parse_ds(train_ds, args.context, args.response_prefix, args.options)
    features = features_generator(train_situations)
    print(features)

main()
