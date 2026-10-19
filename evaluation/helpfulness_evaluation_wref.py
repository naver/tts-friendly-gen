import os
import json
import re
import csv
import time
from collections import defaultdict

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ── Judge prompt ─────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are evaluating whether an assistant's response correctly and helpfully answers a user question.

You will be given:
- A user question (CONTEXT)
- A reference answer (REFERENCE) considered correct and helpful
- A generated answer (GENERATED) to evaluate

Rate the GENERATED answer on a scale from 1 to 5 based SOLELY on factual correctness and helpfulness:

5 - Excellent: Fully and correctly answers the question, consistent with the reference
4 - Good: Mostly correct, minor omissions or slight differences, nothing misleading
3 - Acceptable: Partially answers the question, some inaccuracies or irrelevant additions
2 - Poor: Mostly incorrect or misses the point, significant factual divergence from reference
1 - Bad: Wrong, completely off-topic, or introduces false/harmful information

CRITICAL RULES:
- A response that sounds natural and fluent but gives WRONG or INCOMPLETE information must score LOW
- A response that sounds robotic or unnatural but gives CORRECT and COMPLETE information must score HIGH
- Do NOT reward or penalize based on style, tone, formatting, or how "speech-friendly" the answer sounds
- Only ask: does the generated answer correctly and helpfully respond to the user's question?

CONTEXT: {context}
REFERENCE: {preferred_choice}
GENERATED: {generated_choice}

REASONING: <one sentence explaining your score>
SCORE: <single integer 1-5>"""

ALL_METRICS = ["helpfulness_score"]

def call_judge(context: str, preferred: str, generated: str,
               client: openai.OpenAI, model: str, retries: int = 3):
    prompt = JUDGE_PROMPT.format(
        context=context,
        preferred_choice=preferred,
        generated_choice=generated,
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            text = response.choices[0].message.content.strip()

            score_match = re.search(r'SCORE:\s*([1-5])', text)
            score = int(score_match.group(1)) if score_match else None
            if score is None:
                m = re.search(r'\b([1-5])\b', text)
                score = int(m.group(1)) if m else None

            reasoning = None
            r_match = re.search(r'REASONING:\s*(.+?)(?=\nSCORE:|$)', text, re.DOTALL)
            if r_match:
                reasoning = r_match.group(1).strip()
            else:
                before_score = re.split(r'\nSCORE:', text)[0].strip()
                if before_score:
                    reasoning = before_score

            return score, reasoning

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ⚠️  API error after {retries} attempts: {e}")
                return None, None


# ── File processing ─────────────────────────────────────────

def process_file(filepath: str, client: openai.OpenAI, model: str, skip_sets: tuple[str] = ()):
    """
    Returns:
        split_data: split_name -> metric -> [values]   (for CSV aggregation)
        reasoning_records: list of dicts               (for JSONL output)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_data = defaultdict(lambda: defaultdict(list))
    reasoning_records = []
    total = 0

    for entry in data:
        for split in entry.get("splits", []):
            split_id = split.get("split_id")
            print(f"\n──── split {split_id}", flush=True)
            for set_item in split.get("sets", []):
                split_name = set_item.get("set_id")     # train/validation/test
                if split_name in skip_sets:
                    continue
                print(f"\n────── set {split_name}", flush=True)
                for sample in set_item.get("samples", []):
                    context   = sample.get("context", "")
                    preferred = sample.get("preferred_choice", "")
                    if not preferred:
                        continue
                    for gc in sample.get("generated_choices", []):
                        generated = gc.get("generated_choice", "")
                        if not generated:
                            continue
                        total += 1
                        print(f"  [{total}] Scoring...", end="\r", flush=True)
                        score, reasoning = call_judge(context, preferred, generated, client, model)
                        if score is not None:
                            split_data[split_name]["helpfulness_score"].append(score)
                        reasoning_records.append({
                            "split_id":        split_id,
                            "set_id":          split_name,
                            "context":         context,
                            "preferred_choice": preferred,
                            "generated_choice": generated,
                            "score":           score,
                            "reasoning":       reasoning,
                        })

    print()
    return split_data, reasoning_records


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
         output_file: str = "results/helpfulness_summary.csv",
         model_type: str = None,
         judge_model: str = "gpt-4o-mini",
         skip_sets: tuple[str, ...] = ("train",),
         only_substrings: tuple[str, ...] = None):

    if not HAS_OPENAI:
        print("ERROR: openai package not installed. Run: pip install openai")
        return
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return

    client = openai.OpenAI(api_key=api_key)

    pattern = re.compile(rf"^{re.escape(dataset_prefix)}[\d_]+.*\.json$")
    files = sorted(f for f in os.listdir(directory) if pattern.match(f))

    # Latency files are timing metadata (a dict, not evaluation samples).
    files = [f for f in files if not f.endswith("_latency.json")]

    if model_type:
        files = [f for f in files if model_type in f or "chosen_oracle" in f]

    # When only_substrings is set, restrict to matching file(s) and merge the
    # result into the existing CSV/JSONL instead of overwriting the other rows.
    merge = bool(only_substrings)
    if only_substrings:
        files = [f for f in files if any(s in f for s in only_substrings)]

    print(f"Processing {len(files)} file(s) with judge={judge_model}: {files}", flush=True)
    if not files:
        print("No matching files to process.")
        return

    # Derive reasoning JSONL path from output_file (same dir, same stem)
    reasoning_file = re.sub(r'\.csv$', '_reasoning.jsonl', output_file)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    header1 = ["Approach"]
    for split in ["Train", "Validation", "Test"]:
        header1.extend([split] + [""] * (len(ALL_METRICS) - 1))
    header2 = [""] + ALL_METRICS * 3

    # In merge mode, keep existing CSV rows for files we are NOT re-processing.
    processed = set(files)
    existing_rows = []
    if merge and os.path.exists(output_file):
        with open(output_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            all_rows = list(reader)
        # Skip the two header rows, drop rows we're about to recompute.
        for row in all_rows[2:]:
            if row and row[0] not in processed:
                existing_rows.append(row)

    # In merge mode, keep existing reasoning records for files we are NOT
    # re-processing; otherwise start fresh.
    existing_reasoning = []
    if merge and os.path.exists(reasoning_file):
        with open(reasoning_file, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("source_file") not in processed:
                    existing_reasoning.append(rec)

    new_rows = []
    new_reasoning = []
    for filename in files:
        print(f"\n── {filename}", flush=True)
        split_data, reasoning_records = process_file(
            os.path.join(directory, filename), client, judge_model, skip_sets
        )

        for record in reasoning_records:
            record["source_file"] = filename
            new_reasoning.append(record)

        agg = aggregate(split_data)
        row = [filename]
        for split in ["train", "validation", "test"]:
            for metric in ALL_METRICS:
                row.append(round(agg.get(split, {}).get(metric, 0.0), 4))
        new_rows.append(row)

    rows = sorted(existing_rows + new_rows, key=lambda r: r[0])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header1)
        writer.writerow(header2)
        writer.writerows(rows)

    with open(reasoning_file, "w", encoding="utf-8") as rf:
        for record in existing_reasoning + new_reasoning:
            rf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved → {output_file}")
    print(f"Saved → {reasoning_file}")


if __name__ == "__main__":
    main(
        directory="generation/out",
        dataset_prefix="recipe",
        output_file="evaluation/out/recipe_qwen-4b_helpfulness.csv",
        model_type="Qwen3-4B-Instruct-2507",
    )
    main(
        directory="generation/out",
        dataset_prefix="recipe",
        output_file="evaluation/out/recipe_smollm-3b_helpfulness.csv",
        model_type="SmolLM3-3B",
    )