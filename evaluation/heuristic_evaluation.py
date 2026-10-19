import os
import json
import re
import csv
from collections import defaultdict

# ── Heuristic scoring ──────────────────────────────────────────────

RISK_REGEXES = {
    'acronym_caps':     r'[A-Z]{2,}',
    'numbers':          r'\b\d+(?:[\.,]\d+)?\b',
    'url_like':         r'(https?://|www\.|[A-Za-z0-9]+\.[A-Za-z]{2,})(/\S*)?',
    'mix_alphanum_punct': r'[A-Za-z0-9]+[-_/:.][A-Za-z0-9]+',
    'symbols':          r'[\^\*\+\=\~\|\@\#\$%&]',
    'equation':         r'[A-Za-z]\s*=\s*[^\s]+',
}

WEIGHTS = {
    'acronym_caps':      0.5,
    'numbers':           0.5,
    'url_like':          1.0,
    'mix_alphanum_punct': 0.5,
    'symbols':           0.5,
    'equation':          1.0,
}

def pattern_risk_score(text: str) -> dict:
    L = max(len(text), 20)
    scores = {}
    total = 0.0
    for k, rgx in RISK_REGEXES.items():
        matches = list(re.finditer(rgx, text))
        match_mass = sum(len(m.group(0)) for m in matches)
        s = WEIGHTS[k] * (len(matches) + 0.25 * match_mass / L)
        scores[k] = s
        total += s
    scores['pattern_total'] = total / (1.0 + L / 200.0)
    return scores

def scale_to_1_5(pr: float) -> float:
    return 1.0 + 4.0 * (pr / (pr + 2.0))

def heuristic_score(text: str) -> float:
    """Higher = more TTS-friendly (inverted risk)."""
    pr = pattern_risk_score(text)['pattern_total']
    return 6.0 - scale_to_1_5(pr)

# ── Per-pattern presence metrics ────────

ALL_METRICS = list(RISK_REGEXES.keys()) + ['heuristic_score']

def compute_metrics(text: str) -> dict:
    """
    For each RISK_REGEX: 1 if pattern found, 0 otherwise.
    Also computes the composite heuristic_score (continuous, 1–5).
    """
    metrics = {k: 1 if re.search(rgx, text) else 0
               for k, rgx in RISK_REGEXES.items()}
    metrics['heuristic_score'] = heuristic_score(text)
    return metrics

# ── File processing ─────────────────────────────────────────

def process_file(filepath: str) -> dict:
    """Returns split_name -> metric -> [values]"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_data = defaultdict(lambda: defaultdict(list))

    for entry in data:
        for split in entry.get("splits", []):
            for set_item in split.get("sets", []):
                split_name = set_item.get("set_id")        # train/validation/test
                for sample in set_item.get("samples", []):
                    for gc in sample.get("generated_choices", []):
                        text = gc.get("generated_choice", "")
                        if not text:
                            continue
                        for k, v in compute_metrics(text).items():
                            split_data[split_name][k].append(v)

    return split_data

def aggregate(split_data: dict) -> dict:
    return {
        split: {
            metric: (sum(vals) / len(vals) if vals else 0.0)
            for metric, vals in metrics.items()
        }
        for split, metrics in split_data.items()
    }

# ── Main ────────────────────────────────────────────────────────

def main(directory: str, dataset_prefix: str,
         output_file: str = "results/heuristic_summary.csv",
         model_type: str = None):

    pattern = re.compile(rf"^{re.escape(dataset_prefix)}[\d_]+.*\.json$")
    files = sorted(f for f in os.listdir(directory) if pattern.match(f))

    if model_type:
        files = [f for f in files if model_type in f or "chosen_oracle" in f]

    print(f"Processing {len(files)} file(s): {files}")

    rows = []
    for filename in files:
        if filename.endswith("_latency.json"):
            continue
        split_data = process_file(os.path.join(directory, filename))
        agg = aggregate(split_data)
        row = [filename]
        for split in ["train", "validation", "test"]:
            for metric in ALL_METRICS:
                row.append(round(agg.get(split, {}).get(metric, 0.0), 4))
        rows.append(row)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    header1 = ["Approach"]
    for split in ["Train", "Validation", "Test"]:
        header1.extend([split] + [""] * (len(ALL_METRICS) - 1))

    header2 = [""] + ALL_METRICS * 3

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header1)
        writer.writerow(header2)
        writer.writerows(rows)

    print(f"Saved → {output_file}")


if __name__ == "__main__":
    main(
        directory="generation/out",
        dataset_prefix="cora",
        output_file="evaluation/out/cora_qwen-4b_heuristic.csv",
        model_type="Qwen3-4B-Instruct-2507",
    )
    main(
        directory="generation/out",
        dataset_prefix="cora",
        output_file="evaluation/out/cora_smollm-3b_heuristic.csv",
        model_type="SmolLM3-3B",
    )
    main(
        directory="generation/out",
        dataset_prefix="recipe",
        output_file="evaluation/out/recipe_qwen-4b_heuristic.csv",
        model_type="Qwen3-4B-Instruct-2507",
    )
    main(
        directory="generation/out",
        dataset_prefix="recipe",
        output_file="evaluation/out/recipe_smollm-3b_heuristic.csv",
        model_type="SmolLM3-3B",
    )