import os
import pandas as pd

root_dir = "response_prediction/out"

all_results = []

for approach in os.listdir(root_dir):
    approach_path = os.path.join(root_dir, approach)
    
    if not os.path.isdir(approach_path):
        continue
    
    csv_path = os.path.join(approach_path, "results.csv")
    
    # Skip if file doesn't exist or is empty
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"Skipping {approach} (missing or empty results.csv)")
        continue
    
    try:
        df = pd.read_csv(csv_path, sep="\t")
        print(f"{approach} columns:", df.columns.tolist())
        
        # Add approach name
        df["Approach"] = approach
        
        all_results.append(df)
    
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")

# Combine all results
if not all_results:
    print("No valid results found.")
    exit()

combined_df = pd.concat(all_results, ignore_index=True)

# Normalize column names (optional, if spaces cause issues)
combined_df.columns = [c.strip() for c in combined_df.columns]

# Group by dataset
datasets = combined_df["Dataset"].unique()

output_dir = os.path.join(root_dir, "aggregated_results")
os.makedirs(output_dir, exist_ok=True)

for dataset in datasets:
    dataset_df = combined_df[combined_df["Dataset"] == dataset]
    
    # Reorder columns
    dataset_df = dataset_df[[
        "Approach", "Train Acc", "Validation Acc", "Test Acc"
    ]]
    
    # Sort by performance (optional)
    dataset_df = dataset_df.sort_values(by="Test Acc", ascending=False)
    
    output_path = os.path.join(output_dir, f"{dataset}_results.csv")
    dataset_df.to_csv(output_path, index=False, sep="\t")
    
    print(f"Saved: {output_path}")