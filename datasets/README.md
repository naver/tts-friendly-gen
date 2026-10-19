# CORA and Recipe — TTS-friendly / TTS-unfriendly preference datasets

We introduce two preference datasets in which every record pairs a user question with two answers to that
same question: one written to be read aloud well by a voice assistant (`assistant_chosen`) and
one written to be read aloud badly (`assistant_rejected`). Both datasets are used to train and
evaluate systems that prefer the speech-friendly style.

| | **CORA** | **Recipe** |
|---|---|---|
| domain | coffee ordering assistant | cooking assistant |
| records | 262 | 300 |
| splits (train / val / test) | 100 / 81 / 81 | 100 / 100 / 100 |
| source | synthetic, from a fictional coffee house knowledge base | recipes from [RecipeNLG](https://recipenlg.cs.put.poznan.pl/dataset) |
| completeness of this release | complete | RecipeNLG-derived fields omitted, restored by a script |

All answers (as well as questions for CORA) were produced with `gpt-4o-mini`.

Released with **[Ready to Speak: Aligning LLMs for TTS-Friendly Text Generation](https://arxiv.org/abs/2609.01246)**
— Thibaut Thonet, Jos Rozen, Laurent Besacier (EMNLP 2026; arXiv:2609.01246).

## Files

```
cora/
  json/cora.json                          all 262 records
  json/split-data/{0..4}/train.json       5 random splits
  json/split-data/{0..4}/val.json
  json/split-data/{0..4}/test.json
  json/users.json                         the target user profile
  cora/cora-default/{0..4}/*.parquet      the same splits in Parquet, as Hugging Face datasets

recipe/
  json/recipe_pairs.json                  id + the two answers of all 300 records
  json/split-ids/{0..4}/train.json        the ids of the 5 random splits
  json/split-ids/{0..4}/val.json
  json/split-ids/{0..4}/test.json
  json/users.json                         the target user profile
  restore_recipe_data.py                  restores the omitted files (see below)
```

CORA is released in full and needs no reconstruction. Recipe omits the fields that are verbatim
copies of RecipeNLG; running `restore_recipe_data.py` recreates `json/recipe.json`,
`json/split-data/{0..4}/*.json` and `recipe/recipe-default/{0..4}/*.parquet`, giving Recipe the
same layout as CORA.

The five splits of each dataset are **independent random partitions** of its records, not the
folds of a rotating cross-validation: a record may appear in the test portion of more than one
split. Report results as the mean over the five splits.

The training, preprocessing and evaluation configs consume the Parquet copies and read the
user profile from `json/users.json`.

## Fields

Fields shared by both datasets:

| field | description |
|---|---|
| `human` | the question, e.g. *"Which milk alternatives do you offer for your coffee drinks?"* (CORA) or *"How should I cook Easy Chocolate Chip Muffins (...)"* (Recipe) |
| `assistant_chosen` | the TTS-friendly answer: fractions spelled out, abbreviations expanded, numbers, prices and times verbalised, no URLs or identifiers |
| `assistant_rejected` | the TTS-unfriendly answer: abbreviations, symbols, numerals, URLs and identifiers kept |

CORA also carries:

| field | description |
|---|---|
| `explanation` | free text saying what makes `assistant_rejected` TTS-unfriendly, e.g. *"contains URL 'coffeehouse.com/menu', includes '$0.50'"* |

Recipe also carries:

| field | description |
|---|---|
| `id` | the record's row index in RecipeNLG's `full_dataset.csv` |
| `model` | the model used to produce both answers (`gpt-4o-mini`) |
| `ingredients` | the recipe's ingredient list, copied from RecipeNLG |

In every restored Recipe file — `json/recipe.json`, the `json/split-data` ones and the Parquet
ones alike — `human` is the templated recipe title with the ingredient list appended, which is
the prompt the models are actually trained and evaluated on; `ingredients` additionally keeps the
list on its own, verbatim. The `recipe_pairs.json` released here carries `id`,
`assistant_chosen` and `assistant_rejected` only, and the `split-ids` files carry `id` alone.

`users.json` — the same file in both datasets — holds the single user profile named `default`,
whose `description` is the list of 25 rules (R1–R25) defining what makes a text speech-friendly.
It is the profile used for prompting-based generation.

## Restoring the omitted Recipe fields

The recipe title, which the `human` prompt is built from, and the `ingredients` list are
**verbatim copies of columns of RecipeNLG**, which is distributed under its own click-through
agreement and may not be redistributed by us. They are therefore omitted, and each record keeps
its RecipeNLG `id` so that they can be restored from your own copy of the source corpus:

1. Download `full_dataset.csv` from https://recipenlg.cs.put.poznan.pl/dataset — you will be
   asked to accept RecipeNLG's terms of use, which restrict it to non-commercial research and
   educational purposes.
2. Run:

   ```bash
   python datasets/recipe/restore_recipe_data.py --recipenlg-csv /path/to/full_dataset.csv
   ```

   This writes the restored files in place under `datasets/recipe/`, next to the released ones;
   pass `--output-dir` to write a complete copy of the dataset elsewhere instead. Reconstruction
   is exact — `id` is the row index of `full_dataset.csv` — and the restored fields were verified
   to match our internal files on all 300 records.

## Provenance and terms

CORA is fully synthetic: the questions and both answers were generated with `gpt-4o-mini` from a
fictional coffee house knowledge base (menu, hours, contact details). It contains no third-party data and
no information about real people or businesses.

Recipe's two answers in each record are model rewrites derived from RecipeNLG recipes, and the
questions are templated from RecipeNLG recipe titles. RecipeNLG itself is **not** redistributed
here. Users of Recipe should obtain RecipeNLG directly from the link above and comply with its
terms. Our own use, and this release, are for non-commercial research and educational purposes.

See `LICENSE` and `NOTICE` for the terms applying to this release.

## Citation

If you use either dataset, please cite:

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

If you use Recipe, please also cite its source corpus:

```bibtex
@inproceedings{bien-etal-2020-recipenlg,
  title     = {{R}ecipe{NLG}: A Cooking Recipes Dataset for Semi-Structured Text Generation},
  author    = {Bie{\'n}, Micha{\l} and Gilski, Micha{\l} and Maciejewska, Martyna and
               Taisner, Wojciech and Wisniewski, Dawid and Lawrynowicz, Agnieszka},
  booktitle = {Proceedings of the 13th International Conference on Natural Language Generation},
  year      = {2020},
  pages     = {22--28}
}
```
