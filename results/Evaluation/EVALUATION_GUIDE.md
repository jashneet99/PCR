# Evaluation Guide: SR-NLE + PCR Pipeline

## Overview

This document explains how we evaluate the quality of generated counterfactual explanations
across all 4 models (Qwen, Mistral, Falcon, Llama) and 3 datasets (ComVE, ECQA, eSNLI).

---

## What Are We Evaluating?

At each stage of our pipeline, the model generates a **natural language explanation** for
why a counterfactual sentence is wrong. We measure how similar this generated explanation
is to a **human-written gold explanation**.

```
Example (ComVE):
  Original sentence : "Grizzly bears love honey."
  Counterfactual    : "Grizzly fierce bears hate honey."  [edit_word = "fierce"]

  Gold explanation  : "Honey is good for grizzly bear's growth"
  Init-NLE output   : "Grizzly bears are known to be attracted to the scent of honey..."
  PCR-2 output      : "The word fierce changes the meaning entirely..."
```

---

## Pipeline Stages Evaluated

| Stage | Source File | Field | What It Represents |
|-------|------------|-------|-------------------|
| **Init-NLE** | `explanation_gd.json` | `explanation.final` | Initial explanation before any refinement |
| **SR-NLE** | `iter2_refinement_nl.json` | `nl_refinement.final` | After 2 rounds of SR-NLE feedback + refinement |
| **PCR-1** | `pcr_phase1.json` | `pcr_phase1.de_final` | After PCR Phase 1 (mutual exclusivity constraint) |
| **PCR-2** | `pcr_phase2.json` | `pcr_phase2.de_final` | After PCR Phase 2 (probability margin constraint) |

---

## Gold (Reference) Explanations

Human-written gold explanations come from the original datasets:

| Dataset | Gold File | Total Items |
|---------|----------|-------------|
| ComVE | `data/counterfactual/comve/gen_final.json` | 1000 |
| ECQA | `data/counterfactual/ecqa/gen_final.json` | 1000 |
| eSNLI | `data/counterfactual/esnli/gen_final.json` | 1000 |

> **Important:** We evaluate only on **flipped items** — items where the model changed
> its answer after the counterfactual edit. Each generated item is matched to its
> gold explanation using the `idx` field (ComVE also has it embedded directly).

### Flipped Items per Model

| Dataset | Qwen | Mistral | Falcon | Llama |
|---------|------|---------|--------|-------|
| ComVE | 217 | 660 | 161 | 699 |
| ECQA | 1558 | 1880 | 1655 | 1485 |
| eSNLI | 1715 | 1530 | 1459 | 1304 |

---

## Evaluation Metrics

### 1. BLEU (Bilingual Evaluation Understudy) — Scores 1, 2, 3, 4

BLEU measures **word/phrase overlap** between the generated and gold explanation.

```
Gold : "Honey is good for grizzly bear's growth"
Gen  : "Grizzly bears are known to be attracted to honey"

BLEU-1 counts matching single words  : honey, grizzly, bears → BLEU-1 = 9.09%
BLEU-2 counts matching word pairs    : grizzly bears          → BLEU-2 = 5.40%
BLEU-3 counts matching 3-word groups : (fewer matches)        → BLEU-3 = 3.01%
BLEU-4 counts matching 4-word groups : (even fewer)           → BLEU-4 = 2.03%
```

- Higher n-gram BLEU = stricter similarity check
- BLEU-1 is the most lenient, BLEU-4 is the most strict
- We use **NLTK sentence_bleu** with **smoothing (method1)** to handle zero counts

### 2. ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE-L finds the **Longest Common Subsequence (LCS)** of words between
the generated and gold explanation, preserving word order.

```
Gold : "Honey  is  good  for  grizzly  bear's  growth"
Gen  : "...attracted to honey...grizzly bears..."

LCS  : honey, grizzly  →  ROUGE-L = 13.33%
```

- Unlike BLEU, ROUGE-L does not require consecutive word matches
- We report the **F1 score** (harmonic mean of precision and recall)
- Computed using **rouge_score library** with stemming enabled

### 3. BERTScore (F1)

BERTScore uses a **pre-trained BERT model** to compare the *semantic meaning*
of the generated and gold explanation, rather than just counting exact word matches.

```
Gold : "Honey is good for grizzly bear's growth"
Gen  : "Grizzly bears are naturally attracted to honey"

These sentences use different words but share similar meaning.
BERTScore ≈ 76%  (captures semantic similarity BLEU/ROUGE would miss)
```

- We use **DistilBERT-base-uncased** for efficiency
- Reports **F1 score** (balance of precision and recall at embedding level)
- Range: typically 60–100% for natural language

---

## How Scores Are Aggregated

### Step 1 — Per Item
For each generated explanation, compute BLEU-1/2/3/4, ROUGE-L, BERTScore-F1
against the corresponding gold explanation.

### Step 2 — Per Model (per_model/ folder)
Average scores across all flipped items for that model.

```
comve_qwen_eval.csv   → average over 217 items
comve_mistral_eval.csv → average over 660 items
...
```

### Step 3 — Macro Average (macro_avg/ folder)
Average the per-model scores across all 4 models equally (not weighted by item count).

```
ComVE Init-NLE BLEU-1 = (Qwen:11.32 + Mistral:11.48 + Falcon:10.46 + Llama:8.39) / 4
                       = 10.41
```

### Step 4 — Main Paper Table (main_paper_table.csv)
Combines macro averages for all 3 datasets into one table.

---

## Understanding the Results

### Why Do Scores Decrease from Init-NLE → PCR-2?

This is **expected** and is actually a key finding of our paper.

```
Init-NLE  BLEU-1=10.41  ROUGE-L=17.04  BERT=76.46   ← closest to gold wording
SR-NLE    BLEU-1= 6.86  ROUGE-L=12.06  BERT=74.40
PCR-1     BLEU-1= 7.32  ROUGE-L=12.64  BERT=74.27
PCR-2     BLEU-1= 3.90  ROUGE-L= 7.61  BERT=71.97   ← diverges most from gold
```

**The model is making a deliberate trade-off:**

| Metric | Init-NLE → PCR-2 | Why |
|--------|-----------------|-----|
| BLEU / ROUGE | Decreases | Explanations change wording to focus on the edit word |
| BERTScore | Slight decrease | Meaning shifts toward counterfactual-specific reasoning |
| **Faithfulness** | **Increases** | Edit word appears more often in the explanation |

The gold explanation describes the *original* sentence's logic.
PCR-2 explanations focus on *why the edit word changes the answer* — a different but
more informative style for counterfactual NLE.

This creates a natural tension: **faithfulness ↑ vs. lexical similarity ↓**,
which is the core contribution of our SR-NLE + PCR framework.

---

## Exact File Comparison — What vs What

At every evaluation, exactly **2 files** are compared:

### FILE 1 — Generated Explanation (Model Output)

```
experiments/counterfactual/zs-{dataset}-{model}/{stage_file}.json
```

| Stage | File | Field |
|-------|------|-------|
| Init-NLE | `explanation_gd.json` | `explanation.final` |
| SR-NLE | `iter2_refinement_nl.json` | `nl_refinement.final` |
| PCR-1 | `PCR/experiments/.../pcr_phase1.json` | `pcr_phase1.de_final` |
| PCR-2 | `PCR/experiments/.../pcr_phase2.json` | `pcr_phase2.de_final` |

### FILE 2 — Gold Explanation (Human Written)

```
data/counterfactual/{dataset}/gen_final.json  →  field: gold_explanation
```

So in simple terms:
```
gen_final.json  vs  explanation_gd.json         ← Init-NLE
gen_final.json  vs  iter2_refinement_nl.json    ← SR-NLE
gen_final.json  vs  pcr_phase1.json             ← PCR-1
gen_final.json  vs  pcr_phase2.json             ← PCR-2
```
The **same gold file** is compared against each stage output. ✅

---

## How idx and eidx Are Used for Matching

Each item in the dataset has two identifiers:

- **`idx`** — the original question ID (comes from `gen_final.json`)
- **`eidx`** — the specific edit number for that original question (comes from generated files)

```
gen_final.json (Gold)          Generated file (explanation_gd.json)
──────────────────────         ──────────────────────────────────────────
idx=1 → "Honey is good         idx=1, eidx=0 → edit_word="fierce"  → Gen explanation A
          for grizzly           idx=1, eidx=2 → edit_word="feral"   → Gen explanation B
          bear's growth"        idx=1, eidx=3 → edit_word="instinc" → Gen explanation C
```

**Matching is done ONLY by `idx`:**
- We look up `gold[idx]` from `gen_final.json`
- All 3 edits (eidx=0, 2, 3) of the same original question (idx=1) are each
  compared against the **same gold explanation**
- `eidx` is never used for matching — it only identifies which edit variation it is

This is why total evaluated items > 1000 per model (multiple edits per original question).

---

## Output Files

```
results/Evaluation/
├── REFERENCE_TABLE.md          — quick stage/file mapping reference
├── EVALUATION_GUIDE.md         — this file (full explanation)
├── eval_log.txt                — run log with per-model item counts
├── per_model/                  — 12 CSV files (4 models × 3 datasets)
│   ├── comve_qwen_eval.csv
│   ├── comve_mistral_eval.csv
│   ├── comve_falcon_eval.csv
│   ├── comve_llama_eval.csv
│   ├── ecqa_qwen_eval.csv
│   ├── ecqa_mistral_eval.csv
│   ├── ecqa_falcon_eval.csv
│   ├── ecqa_llama_eval.csv
│   ├── esnli_qwen_eval.csv
│   ├── esnli_mistral_eval.csv
│   ├── esnli_falcon_eval.csv
│   └── esnli_llama_eval.csv
├── macro_avg/                  — 3 CSV files (averaged across 4 models)
│   ├── comve_macro_avg.csv
│   ├── ecqa_macro_avg.csv
│   └── esnli_macro_avg.csv
└── main_paper_table.csv        — final paper table (all datasets + all stages)
```

---

## Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| NLTK | — | BLEU computation with smoothing |
| rouge-score | — | ROUGE-L F1 computation |
| bert-score | — | Semantic similarity via DistilBERT |
| Python | 3.12 | Evaluation script |

---

## Reproducibility

To re-run the full evaluation:
```bash
cd /home/dibyanayan/jashneet/SR-NLE
source sr-nle-env/bin/activate
python3 src/evaluation/run_eval.py
```

Results are saved incrementally — if interrupted, delete the affected CSV and re-run.
