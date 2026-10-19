#!/usr/bin/env python3
"""
2D scatter plot: TTS-friendliness (x) vs Helpfulness (y) per model.

- One point per model per split (val = circle, test = triangle marker)
- Thin dashed line connects val and test points for the same model
- Color = model type
- Hollow markers = *10 dataset variants

Input files are tab-separated with a 2-row header:
  Heuristic file: columns Approach, then per-split sub-columns including heuristic_score
  Helpfulness file: columns Approach, then per-split helpfulness_score

Usage:
  python3 plot_tts_helpfulness_2d.py \
    --heuristic scores_heuristic.tsv \
    --helpfulness scores_helpfulness.tsv \
    --filter cora \
    --output plot_tts_helpfulness.png
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Model type detection ──────────────────────────────────────────────────────

MODEL_TYPES = [
    ("_oracle_", "Oracle", "#F0E442"),  # Yellow
    ("_prompting_",      "Prompting",     "#E69F00"),  # Orange
    ("_sft_",         "SFT",           "#0072B2"),  # Blue
    ("_dpo_",         "DPO",           "#D55E00"),  # Vermillion
    ("_rft-farm_",    "FaST",          "#009E73"),  # Bluish green
    ("_rft-rm_",      "RFT-RM",        "#CC79A7"),  # Reddish purple
    ("_grpo-rm_",     "GRPO-RM",       "#56B4E9"),  # Sky blue
    ("_zeroshot_",    "Zeroshot",      "#000000"),  # Black
]

LEGEND_ORDER = ["Oracle", "Zeroshot", "Prompting", "SFT", "DPO", "GRPO-RM", "RFT-RM", "FaST"]

def get_model_type(filename: str) -> Tuple[str, str]:
    for key, label, color in MODEL_TYPES:
        if key in filename:
            return label, color
    return "Unknown", "#999999"


def is_10fold(name: str) -> bool:
    return "recipe10_" in name or "cora10_" in name


# ── Data loading ──────────────────────────────────────────────────────────────

def parse_two_row_header_tsv(path: str):
    """
    Parse a TSV with a 2-row header like:
        Approach  Train          Validation          Test
                  col1 col2 ...  col1 col2 ...  col1 col2 ...
    Returns (rows, col_map) where col_map maps qualified name -> index,
    e.g. "Train.heuristic_score" -> 3, "Validation.heuristic_score" -> 10
    """
    with open(path, newline="", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    # Find first non-empty line
    row1 = lines[0].split("\t")
    row2 = lines[1].split("\t")

    # Build column map: propagate split names across sub-columns
    col_map: Dict[str, int] = {}
    current_split = ""
    for i, (r1, r2) in enumerate(zip(row1, row2)):
        if r1.strip():
            current_split = r1.strip()
        sub = r2.strip()
        if sub:
            col_map[f"{current_split}.{sub}"] = i

    # Parse data rows
    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        rows.append(line.split("\t"))

    return rows, col_map


def load_heuristic(tsv_path: str, filter_str: Optional[str]) -> Dict[str, Dict]:
    """Load heuristic_score from tab-separated file. Returns {approach: {val, test}}."""
    rows, col_map = parse_two_row_header_tsv(tsv_path)

    val_col = col_map.get("Validation.heuristic_score")
    test_col = col_map.get("Test.heuristic_score")

    if val_col is None or test_col is None:
        raise ValueError(
            f"Could not find heuristic_score columns in {tsv_path}.\n"
            f"Available columns: {list(col_map.keys())}"
        )

    data = {}
    for row in rows:
        if not row or not row[0].strip():
            continue
        name = row[0].strip()
        if filter_str and filter_str not in name:
            continue

        def safe_float(idx):
            try:
                v = row[idx].strip()
                return float(v) if v else None
            except (IndexError, ValueError):
                return None

        data[name] = {
            "heuristic_val": safe_float(val_col),
            "heuristic_test": safe_float(test_col),
        }
    return data


def load_helpfulness(tsv_path: str, filter_str: Optional[str]) -> Dict[str, Dict]:
    """Load helpfulness_score from tab-separated file. Returns {approach: {val, test}}."""
    rows, col_map = parse_two_row_header_tsv(tsv_path)

    val_col = col_map.get("Validation.helpfulness_score")
    test_col = col_map.get("Test.helpfulness_score")

    if val_col is None or test_col is None:
        raise ValueError(
            f"Could not find helpfulness_score columns in {tsv_path}.\n"
            f"Available columns: {list(col_map.keys())}"
        )

    data = {}
    for row in rows:
        if not row or not row[0].strip():
            continue
        name = row[0].strip()
        if filter_str and filter_str not in name:
            continue

        def safe_float(idx):
            try:
                v = row[idx].strip()
                return float(v) if v else None
            except (IndexError, ValueError):
                return None

        data[name] = {
            "helpfulness_val": safe_float(val_col),
            "helpfulness_test": safe_float(test_col),
        }
    return data


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(heuristic: Dict, helpfulness: Dict, output: str, title: str, show_model_legend: bool):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    #fig, ax = plt.subplots(figsize=(8, 6))

    points = []
    all_names = set(heuristic) | set(helpfulness)
    for name in sorted(all_names):
        h = heuristic.get(name, {})
        hp = helpfulness.get(name, {})
        hval = h.get("heuristic_val")
        htest = h.get("heuristic_test")
        helpval = hp.get("helpfulness_val")
        helptest = hp.get("helpfulness_test")
        if hval is None or helpval is None:
            continue
        points.append((name, hval, htest, helpval, helptest))

    if not points:
        print("No data points to plot — check that heuristic and helpfulness data overlap.")
        return

    all_x = [p[1] for p in points] + [p[2] for p in points if p[2] is not None]
    all_y = [p[3] for p in points] + [p[4] for p in points if p[4] is not None]
    margin = 0.1
    xmin = max(1.0, min(all_x) - margin)
    xmax = min(5.0, max(all_x) + margin)
    ymin = max(1.0, min(all_y) - margin)
    ymax = min(5.0, max(all_y) + margin)

    model_legend_handles = []

    for name, hval, htest, helpval, helptest in points:
        label, color = get_model_type(name)
        sname = label #short_name(name)
        hollow = False #is_10fold(name)

        val_kwargs = dict(s=140, zorder=5, marker="o",
                          facecolors="none" if hollow else color,
                          edgecolors=color, linewidths=1.5)
        test_kwargs = dict(s=140, zorder=5, marker="^",
                           facecolors="none" if hollow else color,
                           edgecolors=color, linewidths=1.5)

        ax.scatter(hval, helpval, **val_kwargs)

        if htest is not None and helptest is not None:
            ax.scatter(htest, helptest, **test_kwargs)
            ax.plot([hval, htest], [helpval, helptest],
                    color=color, linewidth=1, linestyle="--", alpha=0.5, zorder=4)

        model_legend_handles.append(
            mlines.Line2D([], [], color=color,
                          marker="o", linestyle="None",
                          markersize=12,
                          markerfacecolor="none" if hollow else color,
                          markeredgecolor=color,
                          label=sname)
        )

    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlabel("TTS-Friendliness (1–5)", fontsize=18)
    ax.set_ylabel("Helpfulness (1–5)", fontsize=18)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

    val_handle = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                               markersize=14, label="Validation")
    test_handle = mlines.Line2D([], [], color="gray", marker="^", linestyle="None",
                                markersize=14, label="Test")

    # Sort legend entries by LEGEND_ORDER (substring match, first match wins).
    # Entries not matching any pattern are appended at the end, alphabetically.
    def legend_sort_key(handle):
        label = handle.get_label()
        for i, pattern in enumerate(LEGEND_ORDER):
            if pattern in label:
                return (i, label)
        return (len(LEGEND_ORDER), label)
 
    model_legend_handles.sort(key=legend_sort_key)

    if show_model_legend:
        print("Show the split legend")
        split_legend = ax.legend(handles=[val_handle, test_handle],
                                     loc="upper left", fontsize=13, title="Split", title_fontsize=13)
        ax.add_artist(split_legend)
        print("Show the model legend")
        ax.legend(handles=model_legend_handles, loc="lower right", ncol=3,
                fontsize=13, title="Model", title_fontsize=13, frameon=True)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output}")
    plt.show()


# Runs that don't require training samples and should appear in both
# the full and 10-sample plots regardless of --filter.
TRAINING_FREE_PATTERNS = ["_oracle_", "_zeroshot_", "_prompting_"]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heuristic", required=True,
                    help="Heuristic scores TSV (2-row header, tab-separated)")
    ap.add_argument("--helpfulness", required=True,
                    help="Helpfulness scores TSV (2-row header, tab-separated)")
    ap.add_argument("--filter", default=None,
                    help="Only include rows whose Approach contains this string "
                         "(training-free runs such as oracle/zeroshot are always included)")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="Exclude rows whose Approach contains any of these strings")
    ap.add_argument("--keep-oracle", action="store_true",
                    help="Include the oracle run (excluded by default)")
    ap.add_argument("--keep-polynorm", action="store_true",
                    help="Include the polynorm run (excluded by default)")
    ap.add_argument("--show-model-legend", action="store_true",
                    help="Show the model legend on the figure")
    ap.add_argument("--output", default="plot_tts_helpfulness_2d.png")
    ap.add_argument("--title", default="TTS-Friendliness vs Helpfulness")
    args = ap.parse_args()

    # Load everything (no filter yet, so we can apply the training-free logic)
    heuristic = load_heuristic(args.heuristic, filter_str=None)
    helpfulness = load_helpfulness(args.helpfulness, filter_str=None)

    def keep(name: str) -> bool:
        """Return True if this row should be included in the plot."""
        # Always exclude oracle unless explicitly kept
        if not args.keep_oracle and "oracle" in name:
            return False
        # Always exclude polynorm unless explicitly kept
        if not args.keep_polynorm and "polynorm" in name:
            return False
        # Training-free runs bypass the --filter check
        if any(p in name for p in TRAINING_FREE_PATTERNS):
            return True
        # Normal filter
        if args.filter and args.filter not in name:
            return False
        return True

    heuristic   = {k: v for k, v in heuristic.items()   if keep(k)}
    helpfulness = {k: v for k, v in helpfulness.items() if keep(k)}

    if args.exclude:
        heuristic = {k: v for k, v in heuristic.items()
                     if not any(e in k for e in args.exclude)}
        helpfulness = {k: v for k, v in helpfulness.items()
                       if not any(e in k for e in args.exclude)}

    print(f"Heuristic data:   {len(heuristic)} models")
    print(f"Helpfulness data: {len(helpfulness)} models")
    overlap = set(heuristic) & set(helpfulness)
    print(f"Overlap (plotted): {len(overlap)} models")
    if overlap != set(heuristic) or overlap != set(helpfulness):
        only_h = set(heuristic) - overlap
        only_hp = set(helpfulness) - overlap
        if only_h:
            print(f"  Only in heuristic (skipped): {sorted(only_h)}")
        if only_hp:
            print(f"  Only in helpfulness (skipped): {sorted(only_hp)}")

    plot(heuristic, helpfulness, args.output, args.title, args.show_model_legend)


if __name__ == "__main__":
    main()