from utils.features import *
from datasets import load_dataset, DatasetDict
import numpy as np
from tqdm import tqdm
import dill as pickle
import argparse
import os
import copy
import csv
import time
import yaml

def pairwise_preprocess(examples, feature_names, context, response_prefix, options):
    feature_names += [response_prefix]
    new_examples = dict()
    new_examples[f"{context}"] = []
    for feature in feature_names:
        new_examples[f"{feature}_chosen"] = []
        new_examples[f"{feature}_rejected"] = []

    for i in range(len(examples[f"{context}"])):
        for option in options:
            if option == "chosen":
                continue # Only consider rejected options to contrast with the chosen option
            new_examples[f"{context}"].append(examples[f"{context}"][i])
            for feature in feature_names:
                new_examples[f"{feature}_chosen"].append(examples[f"{feature}_chosen"][i])
                new_examples[f"{feature}_rejected"].append(examples[f"{feature}_{option}"][i])

    return new_examples

def parse_ds(ds, feature_names, options):
    feature_values = {
        "train": {option: np.zeros((len(ds["train"]), len(feature_names))) for option in options},
        "validation": {option: np.zeros((len(ds["validation"]), len(feature_names))) for option in options},
        "test": {option: np.zeros((len(ds["test"]), len(feature_names))) for option in options}
    }
    preferences = {"train": [], "validation": [], "test": []}
    for split_name in ["train", "validation", "test"]:
        for i, row in enumerate(tqdm(ds[split_name], desc='parsing dataset')):
            for option in options:
                # Extract dataset features
                for j, feature_name in enumerate(feature_names):
                    key = f'{feature_name}_{option}'
                    if key in row:
                        feature_values[split_name][option][i,j] = row[key]
                    else:
                        raise RuntimeError(f"Unknown feature: {key}")
            preferences[split_name].append("chosen")

    return feature_values, preferences

def standardize(feature_values):
    mean = dict()
    std = dict()
    for split_name in ["train", "validation", "test"]:
        all_feature_values = np.concatenate(list(feature_values[split_name].values()), axis=0)
        mean[split_name] = all_feature_values.mean(0)
        std[split_name] = all_feature_values.std(0)+1e-7

    standardized_feature_values = {
        split_name: {option_name: (values - mean[split_name]) / std[split_name] for option_name, values in feature_values[split_name].items()}
        for split_name in ["train", "validation", "test"]
    }

    return standardized_feature_values, mean, std

def get_preferred_option_values(values, preferences, options):
    preferred_option_values = np.zeros(values[options[0]].shape, dtype=values[options[0]].dtype)
    for i, preferred_option in enumerate(preferences):
        preferred_option_values[i] = values[preferred_option][i]
    return preferred_option_values

def get_ebm_scores(options, all_feature_values, coefficients):
    scores = dict()
    for option in options:
        # Compute sum_i lambda_i * phi_i(q, r)
        scores[option] = (coefficients * all_feature_values[option]).sum(1)

    max_log_score = np.max(np.stack([scores[option] for option in options]), axis=0)

    for option in options:
        scores[option] = np.exp((scores[option] - max_log_score).astype(np.longdouble))

    return scores

def get_ebm_moments(ebm_scores, feature_values):
    # Denominator = sum_y exp(sum_i lambda_i * phi_i(q, r))
    z = sum(ebm_scores.values())
    non_zero_z = z > 0
    denominator = z[non_zero_z][..., np.newaxis]
    # denominator: Array(n_samples, 1)

    # Numerator = sum_y exp(sum_i lambda_i * phi_i(q, r)) * phi(q, r)
    numerator = sum((ebm_scores[option][non_zero_z][..., np.newaxis] * feature_values[option][non_zero_z] for option in ebm_scores.keys()))
    # numerator: Array(n_samples, n_features)

    return np.mean(numerator / denominator, 0)

def train_coefficients(target_moments, feature_values, preferences, options, tolerance=1e-1, lr=0.1, max_iter=500, reg_weight=0.0):
    coefficients = np.zeros(len(target_moments))
    iter = 0
    while iter < max_iter:
        # Compute the EBM scores: exp(sum_i lambda_i * phi_i(q, r))
        ebm_scores = get_ebm_scores(options, feature_values, coefficients)
        # ebm_scores: {"chosen": Array(n_samples), "rejected": Array(n_samples)}

        # Compute the EBM moments:
        # 1 / |D| * sum_{q in D)
        #   sum_r exp(sum_i lambda_i * phi_i(q, r)) * phi(q, r) /
        #   sum_r exp(sum_i lambda_i * phi_i(q, r))
        ebm_moments = get_ebm_moments(ebm_scores, feature_values)
        # ebm_moments: Array(n_features)

        # Compute the gradient as target_moments - ebm_moments
        reg_weight_vector = np.full_like(coefficients, reg_weight, dtype=np.float32) # Vector of weights for the L2 regularization loss
        grad = (target_moments - ebm_moments - reg_weight_vector * coefficients).astype(np.float32)

        # Update the lambda coefficients by gradient ascent
        coefficients += lr * grad

        # Compute metrics
        if len(grad) > 0:
            err = np.abs(grad).max()
        else:
            err = np.inf
        acc = calc_acc(ebm_scores, preferences, options)

        if err < tolerance:
            break

        iter += 1
    return coefficients, acc

def calc_loss(ebm_scores, preferences, options):
    # Denominator = sum_y exp(sum_i lambda_i * phi_i(q, r))
    denominator = sum(ebm_scores.values())
    # denominator: Array(n_samples)

    # Numerator = exp(sum_i lambda_i * phi_i(q, r_chosen))
    numerator = get_preferred_option_values(ebm_scores, preferences, options)
    # numerator: Array(n_samples)

    # loss = log(numerator / denominator)
    loss = np.log(numerator) - np.log(denominator)
    # loss: Array(n_samples)

    return loss.mean(), loss.std()

def calc_acc(ebm_scores, preferences, options):
    hits = []
    for i in range(len(preferences)):
        chosen_score = ebm_scores[preferences[i]][i]
        rejected_scores = [ebm_scores[option][i] for option in options if option != preferences[i]]
        max_rejected_score = max(rejected_scores)

        if chosen_score > max_rejected_score: # Success
            hit = 1
        else: # Failure
            hit = 0

        hits.append(hit)

    return np.mean(hits)

def get_model_statistics(feature_values, preferences, coefficients):
    inference_dict = dict()
    options = list(feature_values["train"].keys())
    for split_name in ["train", "validation", "test"]:
        inference_dict[split_name] = dict()

        # Compute the EBM scores: exp(sum_i lambda_i * phi_i(q, r))
        ebm_scores = get_ebm_scores(options, feature_values[split_name], coefficients)

        # Compute the loss:
        # loss = 1 / |D| * sum_{q in D) log(
        #   exp(sum_i lambda_i * phi_i(q, r_chosen)) /
        #   sum_r exp(sum_i lambda_i * phi_i(q, r))
        # )
        loss, loss_std = calc_loss(ebm_scores, preferences[split_name], options)
        inference_dict[split_name]["loss"] = loss
        inference_dict[split_name]["loss_std"] = loss_std
        print(f"{split_name} loss: {loss} +/- {loss_std}")
        acc = calc_acc(ebm_scores, preferences[split_name], options)
        inference_dict[split_name]["acc"] = acc
        print(f"{split_name} Accuracy: {acc}")

    return inference_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='path to YAML config file (if other arguments are not specified)')
    ap.add_argument('--data-dir', default='datasets/cora/cora_cora-auto-5dot4-40_qwen3-4b-instruct-2507_mean')
    ap.add_argument('--options', nargs='+', default=['chosen', 'rejected'])
    ap.add_argument('--context', default='human')
    ap.add_argument('--response-prefix', default='assistant')
    ap.add_argument('--cv', action='store_true', help='to be used if the dataset contains several splits for cross-validation')
    ap.add_argument('--num-cv', default=5, type=int, help='number of cross-validation splits')
    ap.add_argument('--reg-weight', default=0.0, type=float, help='weight for the L2 regularization loss')
    ap.add_argument('--train-size', default="", type=str, help='number of training samples (use "" for all samples)')
    ap.add_argument('--pairwise', action='store_true', help='use pairwise samples (1 chosen and 1 rejected) instead of n-wise (1 chosen and n-1 rejected)')
    args = ap.parse_args()

    # Load YAML if provided
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config_dict = {k.replace("-", "_"): v for k, v in config_dict.items()} # Automatically done in argparse
        for k, v in config_dict.items():
            setattr(args, k, v)
    print(args)

    ################ Prepare data ################

    # Suffix and feature name definition
    prefix = args.data_dir.split("/")[-1]
    prefix_split = prefix.split("_")
    data_type = prefix_split[0]
    feature_type = prefix_split[1]
    model_name = prefix_split[2]
    mode = prefix_split[3]
    prefix = "_".join([f"{data_type}{args.train_size}", feature_type, model_name, mode])
    if model_name == "phi-4-mini-instruct":
        model_name = "microsoft/Phi-4-mini-instruct"
    elif model_name == "llama-3.2-3b-instruct":
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_name == "qwen3-4b-instruct-2507":
        model_name = "Qwen/Qwen3-4B-Instruct-2507"
    elif model_name == "qwen3-1dot7b":
        model_name = "Qwen/Qwen3-1.7B"

    suffix = ''
    if args.pairwise:
        options = ["chosen", "rejected"] # The options are a pair
        suffix += '_pw'
    else:
        options = args.options

    # Fetch the features based on the feature type
    feature_type_upper = feature_type.upper().replace("-", "_")
    features = eval(feature_type_upper)
    feature_names = list(features.keys())

    # Create the directory that will contain the output
    out_path = f'response_prediction/out/{prefix}_farm{suffix}/'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    overall_start_time = time.time()

    results_dict = dict()
    dataset_list = [dir for dir in os.listdir(args.data_dir)]
    dataset_list.sort()
    for dataset in dataset_list:
        results_dict[dataset] = {"train": dict(), "validation": dict(), "test": dict()}

        if args.cv:
            dataset_splits = [dataset + "/data/" + str(i) for i in range(args.num_cv)]
        else:
            dataset_splits = [dataset + "/data"]

        inference_dict = dict() # Will contain the evaluation results for each dataset
        for dataset_split in dataset_splits:
            print("Training on " + dataset_split)
            dataset_split_path = args.data_dir + "/" + dataset_split

            # Load the dataset
            train_ds, validation_ds, test_ds = load_dataset(dataset_split_path, split=[f'train[:{args.train_size}]', 'validation', 'test'])
            ds = DatasetDict({'train': train_ds, 'validation': validation_ds, 'test': test_ds})

            # Define the dataset used for training by duplicating the original dataset
            trainer_ds = copy.deepcopy(ds)

            # Process examples as pairwise (1 chosen and 1 rejected) if needed
            if args.pairwise:
                trainer_ds = trainer_ds.map(
                    lambda e: pairwise_preprocess(e, feature_names, args.context, args.response_prefix, args.options),
                    batched=True,
                    num_proc=4,
                    remove_columns=ds["train"].column_names,
                )

            # For each sample and for the n (chosen, rejected_1, ..., rejected_n-1) responses, fetch the features' values
            # and the preference choice between the n responses
            feature_values, preferences = parse_ds(trainer_ds, feature_names, options)

            # Normalize the feature values by substracting by the mean and dividing by the standard deviation
            feature_values, features_mean, features_std = standardize(feature_values)

            ################ Train the feature-based reward model ################

            # Get data from the training set
            feature_values_, preferences_ = feature_values["train"], preferences["train"]
            # feature_values_: {"chosen": Array(n_samples, n_features), "rejected": Array(n_samples, n_features)}
            # preferences_: List(n_samples)

            # Compute the target moments: 1 / |D| * sum_{q in D) phi(q, r_chosen)
            target_moments = get_preferred_option_values(feature_values_, preferences_, options).mean(0)
            # target_moments: Array(n_features)

            # Obtain the LG feature weights by localized log-likelihood training
            coefficients, acc = train_coefficients(target_moments, feature_values_, preferences_, options, reg_weight=args.reg_weight)
            # coefficients: Array(n_features)

            ################ Evaluate model ################

            # In case of pairwise processing, re-compute feature_values and preferences for the n-wise dataset
            if args.pairwise:
                # For each sample and for the n (chosen, rejected_1, ..., rejected_n-1) responses, get the features' values,
                # the preference choice between the n responses
                feature_values, preferences = parse_ds(ds, feature_names, args.options)

                # Normalize the feature values by substracting by the mean and dividing by the standard deviation
                feature_values, features_mean, features_std = standardize(feature_values)

            inference_dict[dataset_split] = get_model_statistics(feature_values, preferences, coefficients)

            ################ Save parameters and metric values ################

            # Save as a pickle file
            save_path = out_path + dataset_split + "/save.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as pickle_file:
                pickle.dump(
                    {
                        'data_type': data_type,
                        'feature_type': feature_type,
                        'model_name': model_name,
                        'mode': mode,
                        'features': features,
                        'feature_names': feature_names,
                        'features_mean': features_mean,
                        'features_std': features_std,
                        'coefficients': coefficients,
                        "acc": acc,
                        "target_moments": target_moments
                    },
                    pickle_file
                )
            print(f'Parameters saved to {save_path}')

        ################ Aggregate cross-validation results ################

        # Aggregate the results over the different cross-validation splits
        results_dict[dataset]["train"]["loss"] = 1 / len(dataset_splits) * sum([inference_dict[dataset_split]["train"]["loss"] for dataset_split in dataset_splits])
        results_dict[dataset]["validation"]["loss"] = 1 / len(dataset_splits) * sum([inference_dict[dataset_split]["validation"]["loss"] for dataset_split in dataset_splits])
        results_dict[dataset]["test"]["loss"] = 1 / len(dataset_splits) * sum([inference_dict[dataset_split]["test"]["loss"] for dataset_split in dataset_splits])
        results_dict[dataset]["train"]["acc"] = 1 / len(dataset_splits) * sum([inference_dict[dataset_split]["train"]["acc"] for dataset_split in dataset_splits])
        results_dict[dataset]["validation"]["acc"] = 1 / len(dataset_splits) * sum([inference_dict[dataset_split]["validation"]["acc"] for dataset_split in dataset_splits])
        results_dict[dataset]["test"]["acc"] = 1 / len(dataset_splits) * sum([inference_dict[dataset_split]["test"]["acc"] for dataset_split in dataset_splits])

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


main()
