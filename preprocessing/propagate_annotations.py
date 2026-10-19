import datasets
import os
import shutil
import argparse
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='path to YAML config file (if other arguments are not specified)')
    ap.add_argument('--pivot-dir', default='')
    ap.add_argument('--pivot-name', default='cora-default0')
    ap.add_argument('--data-dir', default='')
    ap.add_argument('--suffix', default='_cora-auto-5dot4-40_qwen3-4b-instruct-2507_mean')
    ap.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    ap.add_argument('--context', default='human')
    ap.add_argument('--options', nargs='+', default=['chosen', 'rejected'])
    ap.add_argument('--response-prefix', default='assistant')
    ap.add_argument('--mid-score', default=3, type=float, help='filler mid-range score used in case of missing value')
    ap.add_argument('--num-cv', default=5, type=int, help='the number of cross-validation splits')
    args = ap.parse_args()

    # Load YAML if provided
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config_dict = {k.replace("-", "_"): v for k, v in config_dict.items()} # Automatically done in argparse
        for k, v in config_dict.items():
            setattr(args, k, v)
    print(args)

    ################ Get the feature scores from the annotated pivot data ################

    # Load the annotated pivot dataset
    annotated_data_path = args.pivot_dir + "/" + args.pivot_name + args.suffix + "/data/"
    annotated_data = datasets.load_dataset(annotated_data_path)

    # Get the feature list from the dataset
    feature_list = annotated_data[args.splits[0]].features.keys()
    feature_list = [feature for feature in feature_list if feature.endswith(args.options[0]) and not feature.startswith(args.response_prefix)]
    feature_list = ["_".join(feature.split("_")[:-1]) for feature in feature_list] # Remove the trailing "_chosen" in the feature names

    # Extract mapping between questions and features for the annotated dataset
    annotated_data_dict = dict()
    for split in args.splits:
        annotated_data_split = annotated_data[split]
        for i in range(len(annotated_data_split[args.context])):
            key = annotated_data_split[args.context][i]
            annotated_data_dict[key] = dict()
            for feature in [args.response_prefix] + feature_list:
                for option in args.options:
                    score = annotated_data_split[feature + "_" + option][i]
                    if score is not None:
                        annotated_data_dict[key][feature + "_" + option] = score
                    else:
                        # The feature function failed to return a number
                        # Give an arbitrary score in the middle of the range
                        annotated_data_dict[key][feature + "_" + option] = args.mid_score

    ################ Propagate the feature scores to unannotated user-specific datasets ################

    # Identify the list of user-specific datasets
    unannotated_data_dir = args.data_dir
    unannotated_data_list = [dir for dir in os.listdir(unannotated_data_dir)]
    unannotated_data_list.sort()
    print(unannotated_data_list)

    # Apply the features to the unannotated datasets
    for data_name in unannotated_data_list:
        for i in range(args.num_cv):
            print("Processing", data_name + "/" + str(i))
            unannotated_data_path = args.data_dir + "/" + data_name + "/" + str(i)
            unannotated_data = datasets.load_dataset(unannotated_data_path)
            for split in args.splits:
                unannotated_data_split = unannotated_data[split]
                new_column_dict = dict()
                for feature in feature_list:
                    for option in args.options:
                        new_column_dict[feature + "_" + option] = []
                for j in range(len(unannotated_data_split[args.context])):
                    # Identify the chosen and rejected responses for the current user-specific dataset
                    unannotated_response_dict = {
                        option: unannotated_data_split[args.response_prefix + "_" + option][j]
                        for option in args.options
                    }

                    # Identify the chosen and rejected responses for the pivot on the same question
                    key = unannotated_data_split[args.context][j]
                    feature_values = annotated_data_dict[key]
                    reverse_annotated_response_dict = {
                        feature_values[args.response_prefix + "_" + option]: option
                        for option in args.options
                    }

                    # Use the feature values from the pivot to annotate the unannotated data
                    for feature in feature_list:
                        for option in args.options:
                            # Find the option in the pivot (annotated data) for which the response matches that of the
                            # option in the unannotated data
                            annotated_option = reverse_annotated_response_dict[unannotated_response_dict[option]]
                            new_column_dict[feature + "_" + option].append(feature_values[feature + "_" + annotated_option])

                for feature, feature_values in new_column_dict.items():
                    unannotated_data_split = unannotated_data_split.add_column(feature, feature_values)

                unannotated_data[split] = unannotated_data_split

            # Save the newly annotated dataset
            new_data_path = args.data_dir + args.suffix + "/" + data_name + "/data/" + str(i) + "/"
            os.makedirs(os.path.dirname(new_data_path), exist_ok=True)
            unannotated_data.save_to_disk(new_data_path)

            for split in args.splits:
                os.rename(f"{new_data_path}{split}/data-00000-of-00001.arrow", f"{new_data_path}{split}-00000-of-00001.arrow")
                shutil.rmtree(f"{new_data_path}{split}/")
            os.remove(f"{new_data_path}dataset_dict.json")

main()