from datasets import load_dataset, DatasetDict
import argparse
import os
import random
from tqdm import tqdm
import json
import yaml

def parse_ds(ds, context, response_prefix, options):
    situations = []
    for row in tqdm(ds, desc='parsing dataset'):
        situation = row[f"{context}"]
        actions = [row[f"{response_prefix}_{option}"] for option in options]
        random.shuffle(actions) # To avoid always having the chosen action first
        chosen = row[f"{response_prefix}_chosen"]
        situations.append([situation, actions, chosen])

    return situations

def get_model_outputs(ds, context, response_prefix, options, method_name):
    sets_dict = []
    for set_name in ["train", "validation", "test"]: # Generations are also done on the train set for verifications
        eval_ds = ds[set_name] # Situations for which the generations will be made
        eval_situations = parse_ds(eval_ds, context, response_prefix, options)

        samples_dict = []
        for eval_situation in tqdm(eval_situations, desc='copying chosen generations'):
            situation, actions, chosen_action = eval_situation[0], eval_situation[1], eval_situation[2]
            samples_dict.append(
                {
                    "context": situation,
                    "choices": actions,
                    "preferred_choice": chosen_action,
                    "generated_choices": [
                        {
                            "generator_id": method_name,
                            "generated_choice": chosen_action
                        }
                    ]
                }
            )
        sets_dict.append({"set_id": set_name, "samples": samples_dict})

    return sets_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='path to YAML config file (if other arguments are not specified)')
    ap.add_argument('--model-name', default='gpt-4o-2024-11-20') # Only used to indicate the LLM used for generating responses in original data; no new inference is done
    ap.add_argument('--data-dir', default='')
    ap.add_argument('--options', nargs='+', default=['chosen', 'rejected'])
    ap.add_argument('--context', default='human')
    ap.add_argument('--response-prefix', default='assistant')
    ap.add_argument('--cv', action='store_true', help='to be used if the dataset contains several splits for cross-validation')
    ap.add_argument('--num-cv', default=5, type=int, help='number of cross-validation splits')
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

    short_model_name = args.model_name.split("/")[-1]
    method_name = f"oracle_{short_model_name}"

    ################ Copy chosen generations for each dataset ################

    output_dict = []
    dataset_list = [dir for dir in os.listdir(args.data_dir)]
    dataset_list.sort()
    for dataset in dataset_list:
        splits_dict = []

        if args.cv:
            dataset_splits = [args.data_dir + "/" + dataset + "/" + str(i) for i in range(args.num_cv)]
        else:
            dataset_splits = [args.data_dir + "/" + dataset + "/"]

        for dataset_split in dataset_splits:
            split_id = dataset_split.split("/")[-1] if args.cv else ""
            print("Dataset: " + dataset_split)

            ################ Load the dataset ################

            train_ds, val_ds, test_ds = load_dataset(dataset_split, split=[f'train', 'validation', 'test'])
            ds = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

            ################ Copy chosen generations ################

            sets_dict = get_model_outputs(ds, args.context, args.response_prefix, args.options, method_name)
            splits_dict.append({"split_id": split_id, "sets": sets_dict})

        output_dict.append({"profile": dataset, "splits": splits_dict})

    ################ Save outputs ################

    short_data_name = dataset_list[0].split("-")[0]
    save_fname = f'{short_data_name}_{method_name}'

    out_path = 'generation/out/'
    os.makedirs(os.path.dirname(out_path + save_fname), exist_ok=True)

    out_fname = f'{save_fname}.json'
    with open(out_path + out_fname, "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, ensure_ascii=False, indent=2)
    print(f'Saved to {out_path + out_fname}')

main()
