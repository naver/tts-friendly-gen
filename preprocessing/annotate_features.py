import argparse
from utils.features import *
from utils.common import get_respond, get_feature_scorer
from functools import partial
from datasets import load_dataset
import os
import shutil
from tqdm import tqdm
import torch
import yaml

def clean_huggingface(save_fname):
    os.makedirs(save_fname + "/data", exist_ok=True)
    os.rename(save_fname + "/train/data-00000-of-00001.arrow", save_fname + "/data/train-00000-of-00001.arrow")
    os.rename(save_fname + "/validation/data-00000-of-00001.arrow", save_fname + "/data/validation-00000-of-00001.arrow")
    os.rename(save_fname + "/test/data-00000-of-00001.arrow", save_fname + "/data/test-00000-of-00001.arrow")
    shutil.rmtree(save_fname + "/train/")
    shutil.rmtree(save_fname + "/validation/")
    shutil.rmtree(save_fname + "/test/")
    os.remove(save_fname + "/dataset_dict.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='path to YAML config file (if other arguments are not specified)')
    ap.add_argument('--model-name', default='Qwen/Qwen3-4B-Instruct-2507')
    ap.add_argument('--model-type', default='huggingface', choices=['openai', 'huggingface'])
    ap.add_argument('--dataset', default='')
    ap.add_argument('--options', nargs='+', default=['chosen', 'rejected'])
    ap.add_argument('--context', default='human')
    ap.add_argument('--response-prefix', default='assistant')
    ap.add_argument('--feature-type', default='CORA_AUTO_5DOT4_40')
    ap.add_argument('--mode', default='mean', choices=['gen', 'max', 'mean'])
    ap.add_argument('--dtype', default='torch.bfloat16', choices=['torch.bfloat16', 'torch.float32'])
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

    def score_features(row, feature_scorer, features):
        for option in args.options:
            feature_score_dict = feature_scorer(row[args.context], row[f"{args.response_prefix}_{option}"], features)
            for feature in features.keys():
                row[f'{feature}_{option}'] = feature_score_dict[feature]

        return row

    ################ Data and feature preparation ################

    # Load feature extractor
    respond = get_respond(args.model_name, args.model_type, args.mode, seed, eval(args.dtype))
    dataset_name = args.dataset.split("/")[-1]
    if dataset_name.startswith('cora'):
        data_type = 'cora'
    elif dataset_name.startswith('recipe'):
        data_type = 'recipe'
    else:
        data_type = 'other'
    feature_scorer = get_feature_scorer(respond, data_type)

    # Feature type
    features = eval(args.feature_type)
    feature_type_lower = args.feature_type.lower().replace("_", "-")

    ################ Annotate data ################

    annotate = partial(score_features, feature_scorer=feature_scorer, features=features)

    # Using the dataset of a single user as pivot for all users, given the user-agnostic questionnaire
    list_user_dir = os.listdir(args.dataset)
    list_user_dir.sort()
    pivot_user_dir = list_user_dir[0] # By default use the first user in the user list as pivot
    dict_ds = {f"{args.dataset}/{pivot_user_dir}/0": load_dataset(f"{args.dataset}/{pivot_user_dir}/0")}

    for (data_path, ds) in tqdm(dict_ds.items()):
        dict_ds[data_path] = ds.map(annotate)

    ################ Save feature annotations ################

    for (data_path, ds) in dict_ds.items():
        save_fname = '/'.join(args.dataset.split('/')[:-1]) + '/pivots/' + ''.join(data_path.split('/')[-2:])
        save_fname += "_" + feature_type_lower
        save_fname += "_" + args.model_name.split('/')[-1].lower().replace('.', 'dot')
        save_fname += "_" + args.mode
        ds.save_to_disk(save_fname)
        clean_huggingface(save_fname)
        print(f"Saved to {save_fname}")

main()
