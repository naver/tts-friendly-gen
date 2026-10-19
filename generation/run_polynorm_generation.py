from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import argparse
import os
import random
import time
import torch
from tqdm import tqdm
import json
import yaml

# Few-shot text normalization baseline following PolyNorm (Wong et al., 2025,
# https://arxiv.org/abs/2511.03080, CC BY-NC-ND 4.0). The instruction prompt below is 
# reproduced from the paper's Appendix A. In-context (few-shot) examples are appended 
# as alternating user/assistant turns, and the target unnormalized text is provided as 
# the final user turn (the three-component prompt design described in the paper: 
# instruction prompt, in-context examples, and target unnormalized input). The examples 
# are taken from the PolyNorm-Bench en-US ground truth
# (https://github.com/apple/ml-speech-polynorm-bench, CC BY-NC-ND 4.0).

POLYNORM_SYS_TEMPLATE = (
"""You are an accurate text normalizer for {locale}. Your task is to normalize unstandardized text from the following categories to truly reflect how the real speech is, based on the context:
- Cardinal
- Date
- Decimal
- Ordinal
- Fraction
- Time
- Currency
- Unit (Measure)
- Electronic Address (URL or Email)
- Initialism or Acronym
- ISBN
- Roman Numeral
- Telephone
- Sports Score
- Mathematical Expression
- Symbol
- Abbreviation
- Chemical Formula
- Legal Reference
- Vehicle or Product Code
- Geographic Coordinates
- Version Number
- License Plate or Serial Number
- Musical Notation
- Stock Ticker
- Biological Classification
- Address
- Other unnormalized text

Some important rules:
- When normalizing acronyms, spell out to their full forms for clarity, except when the acronym is a widely recognized and pronounceable name (e.g. "NASA" or "NASCAR"). In those cases, keep the acronym as-is and pronounce it as a word.
- If the acronym combines a letter and a word, split accordingly.
- Convert punctuation that is spoken aloud into words. For example, write 'dot' instead of a period in URLs and emails.
- To ensure clarity, segment compound words, websites and file names into recognizable component words rather than keeping them as a whole word.
- Symbols in a file name should be read as is.
- Common file extensions (.jpeg, .jpg, .txt, etc) should be spoken out. Uncommon file extensions should be spelled out.

Only output the normalized text, without any introduction, preamble, explanation or conclusion."""
)

def get_respond(model_name, model_type, seed):
    if model_type == 'huggingface':
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )

        model_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        ).to("cuda")
        model.eval()

        def respond(messages):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            ).to("cuda")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=1024
                )
            out = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            out = out.rstrip("</s>")

            return out

        return respond
    elif model_type == 'openai':
        # OpenAI client
        client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

        # Define the text completion function
        respond = lambda messages: client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            seed=seed,
            messages=messages
        ).choices[0].message.content.strip('"')

        return respond
    else:
        return None

def load_fewshot_examples(examples_file, num_per_category):
    # Load the PolyNorm few-shot examples, keeping the first num_per_category examples of
    # each category (categories and examples are kept in their order of appearance).
    per_category = {}
    order = []
    with open(examples_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            category = example["category"]
            if category not in per_category:
                per_category[category] = []
                order.append(category)
            if len(per_category[category]) < num_per_category:
                per_category[category].append(example)

    fewshot_examples = []
    for category in order:
        fewshot_examples.extend(per_category[category])
    print(f"Loaded {len(fewshot_examples)} few-shot examples "
          f"({num_per_category} per category over {len(order)} categories)")

    return fewshot_examples

def get_polynorm_normalizer(respond, fewshot_examples, locale):

    # Build the fixed part of the prompt (instruction + in-context examples) once
    sys_text = POLYNORM_SYS_TEMPLATE.format(locale=locale)
    fewshot_messages = [{"role": "system", "content": sys_text}]
    for example in fewshot_examples:
        fewshot_messages.append({"role": "user", "content": example["original_text"]})
        fewshot_messages.append({"role": "assistant", "content": example["normalized_text"]})

    def polynorm_normalizer(text):
        # Append the target unnormalized text as the final user turn
        messages = fewshot_messages + [{"role": "user", "content": text}]

        # Generate the normalized version of the input text
        out = respond(messages)
        return out

    return polynorm_normalizer

def normalize_outputs(polynorm_normalizer, input_dict, source_generator_id, method_name):
    # Iterate over the same nested structure as the zero-shot outputs (profiles -> splits ->
    # sets -> samples) and normalize each generated choice produced by the zero-shot baseline.
    # The wall-clock time of each normalization call is recorded to measure the additional
    # latency introduced by the PolyNorm post-processing step.
    generation_times = []
    output_dict = []
    for profile_entry in input_dict:
        splits_dict = []
        for split_entry in profile_entry["splits"]:
            sets_dict = []
            for set_entry in split_entry["sets"]:
                samples_dict = []
                for sample in tqdm(set_entry["samples"], desc="running polynorm normalizer"):
                    # Retrieve the zero-shot generated choice to normalize
                    source_choice = None
                    for generated_choice in sample["generated_choices"]:
                        if source_generator_id is None or generated_choice["generator_id"] == source_generator_id:
                            source_choice = generated_choice["generated_choice"]
                            break
                    if source_choice is None:
                        raise ValueError(
                            f"No generated choice matching generator_id '{source_generator_id}' "
                            f"was found in the input file."
                        )

                    start_time = time.perf_counter()
                    normalized_choice = polynorm_normalizer(source_choice)
                    generation_times.append(time.perf_counter() - start_time)
                    print("# Context:\n", sample["context"])
                    print("# Zero-shot choice:\n", source_choice)
                    print("# Normalized choice:\n", normalized_choice)
                    samples_dict.append(
                        {
                            "context": sample["context"],
                            "choices": sample["choices"],
                            "preferred_choice": sample["preferred_choice"],
                            "generated_choices": [
                                {
                                    "generator_id": method_name,
                                    "generated_choice": normalized_choice
                                }
                            ]
                        }
                    )
                sets_dict.append({"set_id": set_entry["set_id"], "samples": samples_dict})
            splits_dict.append({"split_id": split_entry["split_id"], "sets": sets_dict})
        output_dict.append({"profile": profile_entry["profile"], "splits": splits_dict})

    return output_dict, generation_times

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='path to YAML config file (if other arguments are not specified)')
    ap.add_argument('--model-name', default='Qwen/Qwen3-4B-Instruct-2507')
    ap.add_argument('--model-type', default='huggingface', choices=['openai', 'huggingface'])
    ap.add_argument('--input-file', help='path to the zero-shot output JSON file to normalize')
    ap.add_argument('--examples-file', default='generation/polynorm/en-US_groundtruth.jsonl', help='path to the PolyNorm few-shot examples (JSONL)')
    ap.add_argument('--num-per-category', default=4, type=int, help='number of few-shot examples to use per category')
    ap.add_argument('--locale', default='English (en-US)', help='locale used in the instruction prompt')
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
    fewshot_examples = load_fewshot_examples(args.examples_file, args.num_per_category)
    polynorm_normalizer = get_polynorm_normalizer(respond, fewshot_examples, args.locale)
    short_model_name = args.model_name.split("/")[-1]
    method_name = f"polynorm_{short_model_name}"

    ################ Load the zero-shot outputs to normalize ################

    with open(args.input_file, encoding="utf-8") as json_file:
        input_dict = json.load(json_file)

    # Infer the zero-shot generator id and short data name from the input file name
    input_fname = os.path.basename(args.input_file)
    short_data_name = input_fname.split("_")[0]
    source_generator_id = os.path.splitext(input_fname)[0][len(short_data_name) + 1:] # e.g. "zeroshot_SmolLM3-3B"
    print(f"Normalizing generated choices from generator '{source_generator_id}'")

    ################ Normalization ################

    output_dict, generation_times = normalize_outputs(polynorm_normalizer, input_dict, source_generator_id, method_name)

    ################ Save outputs ################

    save_fname = f'{short_data_name}_{method_name}'

    out_path = 'generation/out/'
    os.makedirs(os.path.dirname(out_path + save_fname), exist_ok=True)

    out_fname = f'{save_fname}.json'
    with open(out_path + out_fname, "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, ensure_ascii=False, indent=2)
    print(f'Saved to {out_path + out_fname}')

    ################ Save latency measurements ################

    # Report the additional latency introduced by the PolyNorm normalization step. The mean is
    # reported both over all generations and excluding the first one, which for a HuggingFace
    # model includes CUDA warmup / graph compilation and is not representative of steady-state.
    num_generations = len(generation_times)
    total_time = sum(generation_times)
    mean_time = total_time / num_generations if num_generations else 0.0
    warm_times = generation_times[1:] if num_generations > 1 else generation_times
    mean_time_warm = sum(warm_times) / len(warm_times) if warm_times else 0.0
    latency_dict = {
        "method": method_name,
        "model_name": args.model_name,
        "input_file": args.input_file,
        "num_examples_per_category": args.num_per_category,
        "num_generations": num_generations,
        "total_time_s": total_time,
        "mean_time_per_generation_s": mean_time,
        "mean_time_per_generation_excl_first_s": mean_time_warm,
        "min_time_s": min(generation_times) if generation_times else 0.0,
        "max_time_s": max(generation_times) if generation_times else 0.0,
        "generation_times_s": generation_times,
    }
    print(f'PolyNorm normalization latency over {num_generations} generations: '
          f'mean {mean_time:.3f}s (excl. first: {mean_time_warm:.3f}s), total {total_time:.1f}s')

    latency_fname = f'{save_fname}_latency.json'
    with open(out_path + latency_fname, "w", encoding="utf-8") as json_file:
        json.dump(latency_dict, json_file, ensure_ascii=False, indent=2)
    print(f'Saved latency measurements to {out_path + latency_fname}')

main()
