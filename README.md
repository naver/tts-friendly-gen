## Repository description

This repository contains the code and datasets associated to the paper titled **Ready to Speak: Aligning LLMs for TTS-Friendly Text Generation**, published at EMNLP 2026. For more details, the paper is available [here](https://arxiv.org/abs/2609.01246). If you found the code or datasets useful and want to refer to our work, please use the following citation:

```bibtex
@inproceedings{thonet-etal-2026-ready,
  title     = {Ready to Speak: Aligning {LLM}s for {TTS}-Friendly Text Generation},
  author    = {Thonet, Thibaut and Rozen, Jos and Besacier, Laurent},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  eprint    = {2609.01246},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2609.01246}
}
```
---

<div style="border: 1px solid #007acc; padding: 10px; border-radius: 5px; background-color: #f0f8ff;">
  <strong>NEWS:</strong> Our paper has been accepted to <strong>EMNLP 2026 Main Conference</strong> and will be presented in <strong>Budapest, Hungary</strong> from <strong>October 24 to 29, 2026</strong>!
</div>

---

## Data description

The `datasets` folder contains the two preference datasets introduced in the paper, **CORA** (coffee ordering assistant, 262 records) and **Recipe** (cooking assistant, 300 records). In both, every record pairs a user question with two answers to that same question: one written to be read aloud well by a voice assistant (`assistant_chosen`) and one written to be read aloud badly (`assistant_rejected`). Each dataset is provided with five independent random splits (labeled `0` to `4`), both as JSON files and as parquet files directly loadable as Huggingface datasets. A complete description of the datasets -- files, fields, provenance and terms -- is given in [`datasets/README.md`](datasets/README.md).

**Recipe requires a restoration step before use.** CORA is released in full, but Recipe omits the fields that are verbatim copies of [RecipeNLG](https://recipenlg.cs.put.poznan.pl/dataset) -- the recipe title that the `human` prompt is built from, and the `ingredients` list -- since RecipeNLG is distributed under its own click-through agreement and may not be redistributed by us. Each Recipe record keeps its RecipeNLG `id` so that these fields can be restored from your own copy of the source corpus:

1. Download `full_dataset.csv` from https://recipenlg.cs.put.poznan.pl/dataset -- you will be asked to accept RecipeNLG's terms of use, which restrict it to non-commercial research and educational purposes.
2. Run:

   ```bash
   python datasets/recipe/restore_recipe_data.py --recipenlg-csv /path/to/full_dataset.csv
   ```

This recreates `datasets/recipe/json/recipe.json`, `datasets/recipe/json/split-data/{0..4}/*.json` and `datasets/recipe/recipe/recipe-default/{0..4}/*.parquet`, giving Recipe the same layout as CORA. Reconstruction is exact; see the [Restoring the omitted Recipe fields](datasets/README.md#restoring-the-omitted-recipe-fields) section of `datasets/README.md` for details. **None of the commands below can be run on Recipe before this step.**

Once restored, the dataset corresponding to the first splitting (labeled as `0`) of either dataset can be loaded as follows:

```python
from datasets import load_dataset

train_ds, validation_ds, test_ds = load_dataset('datasets/cora/cora/cora-default/0', split=['train', 'validation', 'test'])
```

The columns shared by both Huggingface datasets are the following:

- `human`: the content of the question;
- `assistant_chosen`: the TTS-friendly answer to this question (fractions spelled out, abbreviations expanded, numbers, prices and times verbalised, no URLs or identifiers);
- `assistant_rejected`: the TTS-unfriendly answer to this question (abbreviations, symbols, numerals, URLs and identifiers kept).

CORA additionally carries an `explanation` column, and Recipe an `id`, `model` and `ingredients` column; these are described in `datasets/README.md`.

## How to run the code

In this section, we detail how to run the code, including the preprocessing required by the feature-based reward model, the training of the reward models for TTS-friendliness prediction, the TTS-friendly generation with the FaST approach and baselines, and the evaluation of these different methods. We will provide the commands used to run the different scripts, using the CORA dataset and `Qwen3-4B-Instruct-2507` as the generation backbone as an example. Running the scripts for the Recipe dataset or for the `SmolLM3-3B` backbone instead is straightforward, the only difference is that the config files used should be taken from another directory:

- for the Recipe dataset, replace the `cora` directory of the config paths with `recipe` (e.g., `preprocessing/config/recipe/...`, `generation/config/qwen-4b/recipe/...`);
- for the `SmolLM3-3B` backbone, replace the `qwen-4b` directory of the generation config paths with `smollm-3b` (e.g., `generation/config/smollm-3b/cora/...`).

All commands indicated below should be run from the repository root.

The commands that involve training are given in two variants, corresponding to the two training regimes considered in the paper: the **full training set** (100 samples) and the **low-data regime** (10 samples), the latter using the config files suffixed with `_ts10`. The commands that involve no training (Oracle, Zero-shot, Prompting and PolyNorm) are run only once, as they are common to both regimes.

*Note: Some of the scripts (namely, feature list generation and the helpfulness evaluation) assume an access to GPT-5.4 and GPT-4o-mini through OpenAI's API. These will thus require having the variable `OPENAI_API_KEY` set in your bash environment.*

### Package and version requirements

This code should be run with Python 3.10. The required Python packages, along with their recommended version, are provided in `requirements1.txt` and `requirements2.txt`. The packages in `requirements1.txt` should be installed before those in `requirements2.txt`, i.e., one should run the following commands in this order:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -r requirements1.txt
uv pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
uv pip install --no-build-isolation -r requirements2.txt
```

*Note: Depending on the exact TRL version installed, DPO training under FSDP may fail on an import. If so, in `.venv/lib/python3.10/site-packages/trl/models/utils.py`, replace:*

```python
from torch.distributed.fsdp import FSDPModule
```

*with:*

```python
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:
    FSDPModule = None
```

### Preprocessing

The Feature-aware Reward Model (FaRM), which is the reward model at the core of the FaST approach, relies on features and thus requires some preprocessing steps which are described below.

#### 1. Feature list generation

First, a list of features is directly discovered from the data by prompting a strong LLM (here, `GPT-5.4`) with the training portion of the dataset. This is done by running the following commands (for the CORA dataset):

```bash
python -m preprocessing.generate_features --config preprocessing/config/cora/generate-features_5dot4.yml
python -m preprocessing.generate_features --config preprocessing/config/cora/generate-features_5dot4_ts10.yml
```

The first command discovers the features from the full training set, the second one from the 10 training samples of the low-data regime. Each command prints the resulting feature list to the standard output; the list is then meant to be added to `utils/features.py`, from which it is retrieved through the `feature-type` field of the annotation configs of the next step.

*Note: The feature lists used in the paper are already provided in `utils/features.py` (`CORA_AUTO_5DOT4_40` and `CORA10_AUTO_5DOT4_40` for CORA, `RECIPE_AUTO_5DOT4_40` and `RECIPE10_AUTO_5DOT4_40` for Recipe). This step therefore only needs to be run to rediscover features from scratch; note that its output will differ from ours, as feature generation is not deterministic.*

#### 2. Feature-wise response scoring

Once the list of features has been generated for the dataset of interest, the different responses of the dataset should be scored for each feature. This is done by prompting a small, open LLM (here, `Qwen3-4B-Instruct-2507`). In our code, feature-wise response scoring is done in two steps:

- **Feature score annotation**: We score the responses of a single split of the dataset (the pivot split). Doing this once offers the benefit of efficiency and consistency across splits, and is possible due to the fact that the five splits are partitions of the same set of records.
- **Feature score propagation**: We propagate the response scores to the other splits of the dataset.

##### 2.1. Feature score annotation

The commands to conduct score annotation are the following:

```bash
python -m preprocessing.annotate_features --config preprocessing/config/cora/annotate-features_farm_qwen3-4b-instruct-2507_auto-5dot4-40.yml
python -m preprocessing.annotate_features --config preprocessing/config/cora/annotate-features_farm_qwen3-4b-instruct-2507_10-auto-5dot4-40.yml
```

##### 2.2. Feature score propagation

The propagation of score annotations to the other splits is done by running the following commands:

```bash
python -m preprocessing.propagate_annotations --config preprocessing/config/cora/propagate-annotations_farm_qwen3-4b-instruct-2507_auto-5dot4-40.yml
python -m preprocessing.propagate_annotations --config preprocessing/config/cora/propagate-annotations_farm_qwen3-4b-instruct-2507_10-auto-5dot4-40.yml
```

These commands create the feature-annotated copies of the dataset (e.g., `datasets/cora/cora_cora-auto-5dot4-40_qwen3-4b-instruct-2507_mean`) that the FaRM training script consumes.

### TTS-friendliness prediction

This task consists in measuring the ability of preference/reward models to predict which of the two answers to a question is the TTS-friendly one, on unseen questions. We provide below the commands used to run the experiments for this task, for the reward model FaRM as well as for the standard reward model baseline RM. After running each script, a folder is created in `response_prediction/out`. It will contain a CSV file summarizing the prediction accuracy of the approach on the train, validation and test portions of each split, as well as the reward model learned for each split.

*Note: In the case of FaRM, the feature-wise response scoring steps described previously in the Preprocessing section are required prior to running the following commands.*

- **FaRM**:

```bash
python -m response_prediction.train_farm --config response_prediction/config/cora/train-farm_farm_qwen3-4b-instruct-2507_auto-5dot4-40.yml
python -m response_prediction.train_farm --config response_prediction/config/cora/train-farm_farm_qwen3-4b-instruct-2507_10-auto-5dot4-40_ts10.yml
```

- **RM**:

```bash
python -m response_prediction.train_reward --config response_prediction/config/cora/train-reward_rm_qwen3-4b-instruct-2507.yml
python -m response_prediction.train_reward --config response_prediction/config/cora/train-reward_rm_qwen3-4b-instruct-2507_ts10.yml
```

The accuracies of all the approaches run so far can then be gathered in a single CSV file per dataset, written to `response_prediction/out/aggregated_results`, with:

```bash
python -m response_prediction.aggregate_results
```

### TTS-friendly generation

The TTS-friendly generation task involves generating answers that are suitable for being read aloud by a Text-to-Speech system, for unseen questions. We provide below the commands to run TTS-friendly generation (both fine-tuning, when relevant, and response generation) for the different approaches considered in our experiments. These include the Oracle, the baselines Zero-shot, Prompting, PolyNorm, SFT, DPO, RM with RFT / GRPO, and the FaST approach (i.e., FaRM with RFT). Running each of these approaches will create a JSON file containing the generated responses for each question; it can then be found in `generation/out`.

*Note: In the case of the FaST and RM-based approaches, the FaRM and RM models obtained in the TTS-friendliness prediction step are required to fine-tune the generation models. The reward model paths expected by these approaches are given by the `rm_path` field of their config files.*

- **Oracle** (the reference `assistant_chosen` answers, used as an upper bound):

```bash
python -m generation.run_oracle_generation --config generation/config/qwen-4b/cora/run-oracle-generation_oracle.yml
```

- **Zero-shot** (the backbone prompted with the question alone):

```bash
python -m generation.run_zeroshot_generation --config generation/config/qwen-4b/cora/run-zeroshot-generation_zeroshot.yml
```

- **Prompting** (the backbone prompted with the question and the list of TTS-friendliness rules from `datasets/cora/json/users.json`):

```bash
python -m generation.run_prompting_generation --config generation/config/qwen-4b/cora/run-prompting-generation_prompting.yml
```

- **PolyNorm** (few-shot text normalization of the zero-shot outputs, following [PolyNorm](https://arxiv.org/abs/2511.03080)):

```bash
python -m generation.run_polynorm_generation --config generation/config/qwen-4b/cora/run-polynorm-generation_polynorm.yml
```

*Note: The few-shot examples used by PolyNorm come from PolyNorm-Bench, en-US. The PolyNorm-Bench data is licensed under CC BY-NC-ND 4.0 by Apple and is not redistributed in this repository. Before running the command above, download [`en-US_groundtruth.jsonl`](https://github.com/apple/ml-speech-polynorm-bench/blob/main/polynorm_bench/en-US/en-US_groundtruth.jsonl) from the [PolyNorm benchmark repository](https://github.com/apple/ml-speech-polynorm-bench) and place it in `generation/polynorm/en-US_groundtruth.jsonl` (the default path expected by the `--examples-file` argument). The script keeps the first four examples of each of the 27 normalization categories of that file; this number can be changed with the `--num-per-category` argument.*

*Note: PolyNorm normalizes an existing generation rather than producing one from scratch. The Zero-shot command above must therefore be run first, as its output JSON file is what the `input-file` field of the PolyNorm config points to.*

- **SFT**:

```bash
python -m generation.train_sft --config generation/config/qwen-4b/cora/train-sft_sft.yml
python -m generation.train_sft --config generation/config/qwen-4b/cora/train-sft_sft_ts10.yml
```

- **DPO**:

```bash
python -m generation.train_dpo --config generation/config/qwen-4b/cora/train-dpo_dpo.yml
python -m generation.train_dpo --config generation/config/qwen-4b/cora/train-dpo_dpo_ts10.yml
```

- **RM w/ RFT**:

```bash
python -m generation.train_rft --config generation/config/qwen-4b/cora/train-rft_rm-rft.yml
python -m generation.train_rft --config generation/config/qwen-4b/cora/train-rft_rm-rft_ts10.yml
```

- **RM w/ GRPO**:

```bash
python -m generation.train_grpo --config generation/config/qwen-4b/cora/train-grpo_rm-grpo.yml
python -m generation.train_grpo --config generation/config/qwen-4b/cora/train-grpo_rm-grpo_ts10.yml
```

- **FaST (i.e., FaRM w/ RFT)**:

```bash
python -m generation.train_rft --config generation/config/qwen-4b/cora/train-rft_fast-rft_auto-5dot4-40.yml
python -m generation.train_rft --config generation/config/qwen-4b/cora/train-rft_fast-rft_10-auto-5dot4-40_ts10.yml
```

### Evaluation

The generated responses are evaluated along two complementary axes: their **TTS-friendliness**, measured with a set of heuristics, and their **helpfulness**, measured with an LLM-as-a-judge (in practice, `GPT-4o-mini`). Both evaluation scripts scan all the JSON files found in `generation/out` and write their results as a CSV file in `evaluation/out`.

*Note: Unlike the scripts above, the evaluation scripts take no command-line arguments. The datasets and backbones to evaluate are set in the `__main__` block at the bottom of each script.*

- **Heuristic TTS-friendliness evaluation**:

```bash
python -m evaluation.heuristic_evaluation
```

- **Helpfulness evaluation** (the reference-free judge is used for CORA, the reference-based judge -- which is additionally given the `assistant_chosen` answer -- for Recipe):

```bash
python -m evaluation.helpfulness_evaluation_woref
python -m evaluation.helpfulness_evaluation_wref
```

The two axes can finally be visualized jointly, as in the figures of the paper, with the `plot_2d` script. It produces a scatter plot of TTS-friendliness against helpfulness with one point per approach, written to the path given by `--output`:

```bash
python -m evaluation.plot_2d \
  --heuristic evaluation/out/cora_qwen-4b_heuristic.csv \
  --helpfulness evaluation/out/cora_qwen-4b_helpfulness.csv \
  --filter cora_ \
  --title 'CORA, full training set' \
  --output evaluation/out/figs/plot_tts_helpfulness_cora_qwen-4b.pdf \
  --show-model-legend

python -m evaluation.plot_2d \
  --heuristic evaluation/out/cora_qwen-4b_heuristic.csv \
  --helpfulness evaluation/out/cora_qwen-4b_helpfulness.csv \
  --filter cora10_ \
  --title 'CORA, 10 training samples' \
  --output evaluation/out/figs/plot_tts_helpfulness_cora10_qwen-4b.pdf
```

The `--filter` argument selects the approaches to plot by the prefix of their output file name: `cora_` for the full training set and `cora10_` for the low-data regime (respectively `recipe_` and `recipe10_` for the Recipe dataset).
