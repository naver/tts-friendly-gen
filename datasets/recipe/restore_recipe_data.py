#!/usr/bin/env python3
"""Rebuild the full Recipe dataset from the released ids and preference pairs.

The released files omit every field that is a verbatim copy of RecipeNLG (the
recipe title, which the `human` prompt is built from, and the `ingredients`
list), because RecipeNLG is distributed under its own click-through agreement
and may not be redistributed. Each record keeps its RecipeNLG `id`, so those
fields can be restored from your own copy of the source corpus.

  1. Download `full_dataset.csv` from https://recipenlg.cs.put.poznan.pl/dataset
     (you will be asked to accept the dataset's terms of use).
  2. python datasets/recipe/restore_recipe_data.py --recipenlg-csv /path/to/full_dataset.csv

Reconstruction is exact: `id` is the row index of `full_dataset.csv`.

Released files (inputs):

  json/recipe_pairs.json          id + chosen/rejected response of each recipe
  json/split-ids/<fold>/*.json    ids of the train/val/test splits of each fold
  json/users.json                 the user profile (unchanged, just copied)

Restored files (outputs, written in place under this script's directory):

  json/recipe.json                the complete records, in recipe_pairs order
  json/split-data/<fold>/*.json   the complete records of each split
  recipe/recipe-default/<fold>/*.parquet
                                  the same splits as Hugging Face datasets, as
                                  consumed by the training/evaluation configs

In every output, `human` is the prompt the models are trained and evaluated on,
i.e. the recipe question with the ingredient list appended.
"""
import argparse
import ast
import csv
import json
import os
import shutil

from datasets import Dataset

csv.field_size_limit(10 ** 9)

FOLDS = range(5)
PARTS = ("train", "val", "test")

# The `human` prompt of a record is the recipe title of RecipeNLG, wrapped in
# this template; `model` is the LLM that generated the chosen responses.
PROMPT_TEMPLATE = "How should I cook {title}"
INGREDIENTS_TEMPLATE = "\nHere is the list of ingredients for this recipe:\n{ingredients}"
MODEL = "gpt-4o-mini"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def read_recipenlg(csv_path, wanted):
    """id -> (title, ingredients), read straight from the RecipeNLG CSV."""
    found = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or len(header) < 3 or header[1] != "title":
            raise SystemExit(
                f"{csv_path} does not look like RecipeNLG's full_dataset.csv "
                f"(expected columns: '', title, ingredients, directions, ...)"
            )
        for row in reader:
            if not row or not row[0]:
                continue
            try:
                rid = int(row[0])
            except ValueError:
                continue
            if rid in wanted:
                found[rid] = (row[1], ast.literal_eval(row[2]) if row[2] else [])
                if len(found) == len(wanted):
                    break
    missing = wanted - set(found)
    if missing:
        raise SystemExit(
            f"{len(missing)} ids were not found in {csv_path}, e.g. {sorted(missing)[:5]}. "
            f"Make sure this is the full RecipeNLG corpus (2.2M rows), not a subset."
        )
    return found


def build_prompt(title, ingredients):
    """The prompt of a record: the recipe question, then the ingredient list."""
    # A few RecipeNLG titles and ingredients are padded with spaces; those are
    # kept verbatim in the `ingredients` field but stripped in the prompt.
    return PROMPT_TEMPLATE.format(title=title).rstrip() + INGREDIENTS_TEMPLATE.format(
        ingredients="\n".join("- " + ingredient.strip() for ingredient in ingredients)
    )


def build_records(pairs, recipenlg):
    """The released pairs, completed with the fields restored from RecipeNLG."""
    records = []
    for pair in pairs:
        title, ingredients = recipenlg[pair["id"]]
        records.append({
            "id": pair["id"],
            "human": build_prompt(title, ingredients),
            "model": MODEL,
            "assistant_chosen": pair["assistant_chosen"],
            "assistant_rejected": pair["assistant_rejected"],
            "ingredients": ingredients,
        })
    return records


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipenlg-csv", required=True, help="path to RecipeNLG full_dataset.csv")
    p.add_argument("--dataset-dir", default=here, help="directory holding the released files (default: this script's directory)")
    p.add_argument("--output-dir", default=None, help="where to write the restored files (default: --dataset-dir, i.e. in place)")
    args = p.parse_args()
    output_dir = args.output_dir or args.dataset_dir

    pairs_path = os.path.join(args.dataset_dir, "json", "recipe_pairs.json")
    if not os.path.exists(pairs_path):
        raise SystemExit(f"{pairs_path} not found; --dataset-dir should point to the released recipe folder")
    pairs = read_json(pairs_path)

    splits = {}
    for fold in FOLDS:
        for part in PARTS:
            path = os.path.join(args.dataset_dir, "json", "split-ids", str(fold), part + ".json")
            if not os.path.exists(path):
                raise SystemExit(f"{path} not found; the released split ids are incomplete")
            splits[(fold, part)] = [entry["id"] for entry in read_json(path)]

    print(f"{len(pairs)} recipes to restore from {args.recipenlg_csv}")
    recipenlg = read_recipenlg(args.recipenlg_csv, {pair["id"] for pair in pairs})
    records = build_records(pairs, recipenlg)
    by_id = {record["id"]: record for record in records}

    unknown = {rid for ids in splits.values() for rid in ids} - set(by_id)
    if unknown:
        raise SystemExit(f"{len(unknown)} split ids are missing from {pairs_path}, e.g. {sorted(unknown)[:5]}")

    write_json(os.path.join(output_dir, "json", "recipe.json"), records)
    written = 1
    for (fold, part), ids in splits.items():
        split_records = [by_id[rid] for rid in ids]
        write_json(os.path.join(output_dir, "json", "split-data", str(fold), part + ".json"), split_records)

        parquet_path = os.path.join(output_dir, "recipe", "recipe-default", str(fold), part + ".parquet")
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        Dataset.from_list(split_records).to_parquet(parquet_path)
        written += 2

    if os.path.abspath(output_dir) != os.path.abspath(args.dataset_dir):
        # Carry the released files over, so that the output is a complete dataset.
        for rel in [os.path.join("json", name) for name in ("recipe_pairs.json", "users.json")] + [
            os.path.join("json", "split-ids", str(fold), part + ".json") for fold in FOLDS for part in PARTS
        ]:
            destination = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(os.path.join(args.dataset_dir, rel), destination)
            written += 1

    print(f"wrote {written} files to {output_dir}")


if __name__ == "__main__":
    main()
