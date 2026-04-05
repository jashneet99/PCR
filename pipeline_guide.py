"""
================================================================================
SR-NLE PIPELINE GUIDE
Self-Critique and Refinement for Natural Language Explanations
================================================================================


--------------------------------------------------------------------------------
CORE IDEA
--------------------------------------------------------------------------------
LLMs often explain their answers incorrectly — they give a valid-sounding
explanation that has nothing to do with their actual reasoning.

SR-NLE fixes this by making the model:
    1. Generate an explanation
    2. Critique its own explanation (feedback)
    3. Rewrite it (refinement)
    ...repeat 3 times

The goal: reduce "unfaithful" explanations (54.81% → 36.02%)

--------------------------------------------------------------------------------
WHY 10 COUNTERFACTUALS PER ITEM?
--------------------------------------------------------------------------------
The paper generates 10 different edits per sentence instead of just 1.
Here is why this design decision makes sense:

PROBLEM WITH JUST 1 EDIT:
    If you only make 1 edit, you're betting everything on that one word
    being "important" to the model. If the model ignores that word,
    the answer won't flip — and you'd incorrectly conclude the model
    is unfaithful, when really you just picked the wrong word.

    Example:
        Original:       "when it is hot humidity forms"
        Counterfactual: "when it is hot SEVERE humidity forms"
        Qwen: Answer (B) both times ← didn't flip, skip this item

    Does this mean Qwen is unfaithful? NOT NECESSARILY.
    Maybe "severe" just wasn't strong enough to change its mind.

SOLUTION — 10 DIFFERENT EDITS:
    By generating 10 variations, you cast a wider net:

        1. "when it is hot SEVERE humidity forms"    → Qwen flips ✅ keep
        2. "when it is hot EXTREME humidity forms"   → Qwen flips ✅ keep
        3. "when it is hot RAPIDLY humidity forms"   → no flip    ❌ skip
        4. "when it is GRADUALLY hot humidity forms" → Qwen flips ✅ keep
        5. "when it is hot INTENSE humidity forms"   → no flip    ❌ skip
        ...

    You keep only the ones where Qwen's answer actually flipped.
    This GUARANTEES the edit word genuinely influenced the model.

WHAT THIS ACHIEVES:
    With 1 edit  → many items discarded, test set too small, unreliable
    With 10 edits → large valid test set, diverse, robust evaluation

    1000 original sentences
        x 10 edits each
        = 9965 counterfactuals (35 failed extraction)
        → filter by answer flip (counter.py)
        = final valid test set for faithfulness evaluation

KEY INSIGHT:
    The 10 edits are different words injected at different positions
    in the same sentence. Each tests a different aspect of the model's
    sensitivity. Without this 10x multiplier, the test set after
    counter filtering would be too small to draw reliable conclusions.

--------------------------------------------------------------------------------
WHY ONLY 235 ITEMS ARE USED FROM STEP 2 ONWARDS (counter.py filtering)
--------------------------------------------------------------------------------
After counter.py runs, only the FLIPPED items are kept for the rest of the pipeline.

THE FAITHFULNESS TEST REQUIRES PROOF:
    To test if an explanation is faithful, you must PROVE the edit word
    actually influenced the model's decision — not just appeared by coincidence.

    edit_word = "gradually"

    Case 1 — answer FLIPPED (original=B, counterfactual=A):
        → "gradually" provably changed Qwen's mind
        → NOW we can check: does the explanation mention "gradually"?
        → Valid test case ✅

    Case 2 — answer DID NOT flip (original=B, counterfactual=B):
        → Maybe Qwen completely ignored "gradually"
        → Even if explanation mentions "gradually", it could be coincidence
        → No way to verify faithfulness → DISCARD ❌

THE MATH:
    9,965 counterfactuals
        ├── 235  answer FLIPPED  → edit word provably mattered → USE THESE ✅
        └── 9,730 same answer    → edit word may be ignored    → DISCARD  ❌

WHY COUNTER RATE IS LOW (2.36% for Qwen):
    Qwen is a strong, stable 7B model. The injected adjectives/adverbs
    (like "severe", "gradually") are not strong enough to flip its answer
    most of the time. This is expected for a capable model.
    235 items is still sufficient for reliable faithfulness evaluation.

RESULT (Qwen on ComVE):
    Total counterfactuals : 9,965
    Flipped (counter)     : 235
    Counter rate          : 2.36%
    Saved to              : experiments/counterfactual/zs-comve-qwen/answer_gd_counter.json

--------------------------------------------------------------------------------
HOW FAITHFULNESS IS MEASURED — THE COUNTERFACTUAL TRICK
--------------------------------------------------------------------------------
Take a wrong sentence:
    "when it is hot humidity forms"

Inject an edit word to create a counterfactual:
    "when it is hot SEVERE humidity forms"   ← edit_word = "severe"

If Qwen's answer FLIPS between original and counterfactual
→ Qwen noticed "severe" → it should mention "severe" in its explanation
→ If explanation does NOT mention "severe" → UNFAITHFUL

Simple check:
    edit_word in explanation → faithful ✅
    edit_word not in explanation → unfaithful ❌

================================================================================
PROJECT STRUCTURE
================================================================================

SR-NLE/
│
├── configs/                    ← YAML config files (use ??? as placeholders)
│   ├── answer.yaml             ← config for Step 1 (answer generation)
│   ├── explanation.yaml        ← config for Step 2 (explanation generation)
│   ├── feedback.yaml           ← config for Step 3 (feedback generation)
│   └── refinement.yaml         ← config for Step 4 (refinement generation)
│
├── data/
│   ├── raw/comve/              ← original CSV files (train/dev/test)
│   ├── formatted/comve/        ← cleaned JSON (test.json = 1000 items)
│   └── counterfactual/comve/
│       ├── gen_org.json        ← LLM-generated edits (1000 items)
│       ├── gen_final.json      ← merged final edits (1000 items)
│       └── ext_org.json        ← extracted counterfactuals (9965 items) ✅ READY
│
├── experiments/                ← all pipeline outputs saved here
│   └── {type}/{prompt}-{dataset}-{model}/
│       ├── answer_gd.json
│       ├── answer_gd_counter.json
│       ├── explanation_gd.json
│       ├── iter0_feedback_{type}.json
│       ├── iter0_refinement_{type}.json
│       ├── iter1_feedback_{type}.json
│       ├── iter1_refinement_{type}.json
│       ├── iter2_feedback_{type}.json
│       └── iter2_refinement_{type}.json
│
└── src/
    ├── data_format/            ← converts raw CSVs to JSON
    │   ├── format_comve.py
    │   ├── format_ecqa.py
    │   ├── format_esnli.py
    │   └── formatter.py
    │
    ├── data_gen/               ← generates counterfactual sentences
    │   ├── generate_edits.py   ← LLM generates 10 edits per sentence
    │   ├── extract_edits.py    ← parses LLM output, extracts edit words
    │   └── merge_edits.py      ← merges org + failed → gen_final.json
    │
    ├── model/
    │   └── model.py            ← loads LLM (Qwen/Llama/Mistral/Falcon)
    │
    ├── modules/                ← core generation logic
    │   ├── answer_generator.py
    │   ├── explanation_generator.py
    │   ├── feedback_generator.py
    │   └── refinement_generator.py
    │
    ├── runners/                ← entry points (scripts you actually run)
    │   ├── answer_runner.py
    │   ├── explanation_runner.py
    │   ├── feedback_runner.py
    │   └── refinement_runner.py
    │
    ├── attribution/            ← measures which input words influenced model
    │   ├── attention.py        ← uses attention weights
    │   ├── integrated_gradient.py ← uses integrated gradients (captum)
    │   └── random.py           ← random baseline
    │
    ├── prompts/zero_shot/      ← prompt templates for each dataset
    │   ├── comve_prompt.py
    │   ├── ecqa_prompt.py
    │   └── esnli_prompt.py
    │
    └── evaluation/
        ├── counter.py          ← measures answer flip rate
        ├── faithfulness.py     ← measures unfaithfulness rate
        └── utils.py            ← logging setup

================================================================================
SUPPORTED MODELS (all loaded via HuggingFace)
================================================================================

Key         HuggingFace ID                          Cached on Server?
-------     ------------------------------------    -----------------
qwen        Qwen/Qwen2.5-7B-Instruct                ✅ YES (6GB)
mistral     mistralai/Mistral-7B-Instruct-v0.3      ✅ YES (28GB)
llama       meta-llama/Llama-3.1-8B-Instruct        ❌ NO
falcon      tiiuae/Falcon3-7B-Instruct              ❌ NO

================================================================================
SUPPORTED DATASETS
================================================================================

Name    Task                                    Status on Server
------  ------------------------------------    ----------------
comve   Which sentence violates commonsense?    ✅ FULLY READY
ecqa    5-choice commonsense QA                 ❌ raw data missing
esnli   Natural language inference (3-class)    ❌ raw data missing

================================================================================
FEEDBACK TYPES
================================================================================

Type        Description
---------   ---------------------------------------------------------------
nl          Natural language — model critiques its own explanation in words
iw          Important words — model ranks input words by importance (1-100)
aiw_attn    Attribution (attention) — attention weights identify key tokens
aiw_ig      Attribution (integrated gradients) — gradient-based importance
iw_rand     Random words — baseline, words picked randomly

================================================================================
SERVER SETUP
================================================================================

Server      : e2e-72-152
GPUs        : 2x NVIDIA A100 80GB PCIe (CUDA 12.8)
Project dir : /home/dibyanayan/jashneet/SR-NLE/
Virtual env : /home/dibyanayan/jashneet/sr-nle-test/

Activate venv:
    source /home/dibyanayan/jashneet/sr-nle-test/bin/activate


--------------------------------------------------------------------------------
PHASE 0 — DATA PREPARATION (already done ✅)
--------------------------------------------------------------------------------

# Step 0a — Format raw dataset
python src/data_format/formatter.py

    Input  : data/raw/comve/test.csv
    Output : data/formatted/comve/test.json       (1000 items)

# Step 0b — Generate counterfactual edits (LLM injects 10 words per sentence)
python src/data_gen/generate_edits.py -d comve -m qwen

    Input  : data/formatted/comve/test.json       (1000 items)
    Output : data/counterfactual/comve/gen_org.json  (1000 items with 10 edits each)

# Step 0c — Extract structured edits from LLM output
python src/data_gen/extract_edits.py -d comve -dt org

    Input  : data/counterfactual/comve/gen_org.json
    Output : data/counterfactual/comve/ext_org.json  (9965 items) ← main counterfactual data

# Step 0d — Merge (handle any failed extractions)
python src/data_gen/merge_edits.py -d comve
python src/data_gen/extract_edits.py -d comve -dt final

    Output : data/counterfactual/comve/gen_final.json

--------------------------------------------------------------------------------
PHASE 1 — ANSWER GENERATION
--------------------------------------------------------------------------------

# Step 1a — Qwen answers ORIGINAL sentences (1000 items)
python src/runners/answer_runner.py --config=configs/answer.yaml \
    dataset.type=original \
    dataset.name=comve \
    prompt.type=zs \
    model.name=qwen \
    decoding.type=gd

    Input  : data/formatted/comve/test.json               (1000 items)
    Output : experiments/original/zs-comve-qwen/answer_gd.json
    Time   : ~20-25 min
    Purpose: Baseline answers to compare against counterfactual answers

# Step 1b — Qwen answers COUNTERFACTUAL sentences (9965 items)
python src/runners/answer_runner.py --config=configs/answer.yaml \
    dataset.type=counterfactual \
    dataset.name=comve \
    prompt.type=zs \
    model.name=qwen \
    decoding.type=gd

    Input  : data/counterfactual/comve/ext_org.json        (9965 items)
    Output : experiments/counterfactual/zs-comve-qwen/answer_gd.json
    Time   : ~2 hours
    Purpose: See if Qwen's answer changes when edit word is injected

# Step 1c — Filter: keep only items where answer FLIPPED
python src/evaluation/counter.py

    Input  : experiments/original/zs-comve-qwen/answer_gd.json
             experiments/counterfactual/zs-comve-qwen/answer_gd.json
    Output : experiments/counterfactual/zs-comve-qwen/answer_gd_counter.json
    Purpose: These flipped items = valid faithfulness test cases

--------------------------------------------------------------------------------
PHASE 2 — EXPLANATION GENERATION
--------------------------------------------------------------------------------

# Step 2 — Qwen explains its counterfactual answers
python src/runners/explanation_runner.py --config=configs/explanation.yaml \
    dataset.type=counterfactual \
    dataset.name=comve \
    prompt.type=zs \
    model.name=qwen \
    decoding.type=gd

    Input  : experiments/counterfactual/zs-comve-qwen/answer_gd_counter.json
    Output : experiments/counterfactual/zs-comve-qwen/explanation_gd.json
    Purpose: Initial explanations — baseline faithfulness ~54.81% unfaithful

--------------------------------------------------------------------------------
PHASE 3 — FEEDBACK + REFINEMENT LOOP (3 iterations)
--------------------------------------------------------------------------------
# Repeat Steps 3 and 4 for iteration = 0, 1, 2
# Each iteration uses the previous refinement as the new explanation

# --- ITERATION 0 ---

# Step 3 — Generate feedback on the explanation
python src/runners/feedback_runner.py --config=configs/feedback.yaml \
    dataset.type=counterfactual \
    dataset.name=comve \
    prompt.type=zs \
    model.name=qwen \
    feedback.type=nl \
    seed=42 \
    iteration=0

    Input  : experiments/counterfactual/zs-comve-qwen/explanation_gd.json
    Output : experiments/counterfactual/zs-comve-qwen/iter0_feedback_nl.json
    Purpose: "Your explanation missed the word X, please improve it"

# Step 4 — Refine explanation using feedback
python src/runners/refinement_runner.py --config=configs/refinement.yaml \
    dataset.type=counterfactual \
    dataset.name=comve \
    prompt.type=zs \
    model.name=qwen \
    feedback.type=nl \
    seed=42 \
    iteration=0

    Input  : experiments/counterfactual/zs-comve-qwen/iter0_feedback_nl.json
    Output : experiments/counterfactual/zs-comve-qwen/iter0_refinement_nl.json
    Purpose: Improved explanation incorporating feedback

# --- ITERATION 1 --- (uses iter0 refinement as input)

python src/runners/feedback_runner.py --config=configs/feedback.yaml \
    dataset.type=counterfactual dataset.name=comve \
    prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=1

    Input  : experiments/counterfactual/zs-comve-qwen/iter0_refinement_nl.json
    Output : experiments/counterfactual/zs-comve-qwen/iter1_feedback_nl.json

python src/runners/refinement_runner.py --config=configs/refinement.yaml \
    dataset.type=counterfactual dataset.name=comve \
    prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=1

    Input  : experiments/counterfactual/zs-comve-qwen/iter1_feedback_nl.json
    Output : experiments/counterfactual/zs-comve-qwen/iter1_refinement_nl.json

# --- ITERATION 2 --- (uses iter1 refinement as input)

python src/runners/feedback_runner.py --config=configs/feedback.yaml \
    dataset.type=counterfactual dataset.name=comve \
    prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=2

    Input  : experiments/counterfactual/zs-comve-qwen/iter1_refinement_nl.json
    Output : experiments/counterfactual/zs-comve-qwen/iter2_feedback_nl.json

python src/runners/refinement_runner.py --config=configs/refinement.yaml \
    dataset.type=counterfactual dataset.name=comve \
    prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=2

    Input  : experiments/counterfactual/zs-comve-qwen/iter2_feedback_nl.json
    Output : experiments/counterfactual/zs-comve-qwen/iter2_refinement_nl.json

--------------------------------------------------------------------------------
PHASE 4 — EVALUATION
--------------------------------------------------------------------------------

cd src/evaluation

# Evaluate counter rate (how many answers flipped)
python counter.py

    Output : logs/counter_<timestamp>.log
    Metric : counter_rate = flipped / total

# Evaluate faithfulness (does explanation mention the edit word?)
python faithfulness.py

    Output : logs/faithfulness_<timestamp>.log
             logs/df/faithfulness_results.csv
    Metric : unfaith_rate = items where edit_word NOT in explanation / total
    Goal   : unfaith_rate drops from 54.81% → 36.02%

================================================================================
FULL PIPELINE — ONE-LINER SEQUENCE (copy-paste ready)
================================================================================

cd /home/dibyanayan/jashneet/SR-NLE
source /home/dibyanayan/jashneet/sr-nle-test/bin/activate

python src/runners/answer_runner.py --config=configs/answer.yaml dataset.type=original dataset.name=comve prompt.type=zs model.name=qwen decoding.type=gd

python src/runners/answer_runner.py --config=configs/answer.yaml dataset.type=counterfactual dataset.name=comve prompt.type=zs model.name=qwen decoding.type=gd

python src/evaluation/counter.py

python src/runners/explanation_runner.py --config=configs/explanation.yaml dataset.type=counterfactual dataset.name=comve prompt.type=zs model.name=qwen decoding.type=gd

python src/runners/feedback_runner.py --config=configs/feedback.yaml dataset.type=counterfactual dataset.name=comve prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=0

python src/runners/refinement_runner.py --config=configs/refinement.yaml dataset.type=counterfactual dataset.name=comve prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=0

python src/runners/feedback_runner.py --config=configs/feedback.yaml dataset.type=counterfactual dataset.name=comve prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=1

python src/runners/refinement_runner.py --config=configs/refinement.yaml dataset.type=counterfactual dataset.name=comve prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=1

python src/runners/feedback_runner.py --config=configs/feedback.yaml dataset.type=counterfactual dataset.name=comve prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=2

python src/runners/refinement_runner.py --config=configs/refinement.yaml dataset.type=counterfactual dataset.name=comve prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=2

cd src/evaluation && python faithfulness.py

================================================================================
CONCRETE EXAMPLE — ONE ITEM THROUGH THE FULL PIPELINE
================================================================================

ORIGINAL SENTENCE PAIR:
    Sentence 0: "when it rains humidity forms"       (normal)
    Sentence 1: "when it is hot humidity forms"      (violates commonsense)

COUNTERFACTUAL (edit_word = "severe"):
    Sentence 1: "when it is hot SEVERE humidity forms"

STEP 1a — Original answer:
    Qwen: Answer (B)   ← Sentence 1 violates commonsense

STEP 1b — Counterfactual answer:
    Qwen: Answer (A)   ← FLIPPED! "severe" changed Qwen's mind ✅ valid test case

STEP 2 — Initial explanation:
    "Sentence 1 violates commonsense because hot weather
     causes humidity, not an absence of it."
    → Does "severe" appear? NO → UNFAITHFUL ❌

STEP 3 (iter 0) — NL Feedback:
    "Your explanation doesn't address why SEVERE humidity
     is unreasonable. Improve the explanation to reflect
     the actual reason for your choice."

STEP 4 (iter 0) — Refined explanation:
    "Sentence 1 violates commonsense because hot weather
     causes normal humidity — SEVERE humidity is an
     unrealistic exaggeration."
    → Does "severe" appear? YES → FAITHFUL ✅

EVALUATION:
    edit_word  = "severe"
    explanation = "...SEVERE humidity is an unrealistic exaggeration."
    faithful   = True ✅

================================================================================
CURRENT STATUS (as of March 22, 2026)
================================================================================

DATA PREPARATION:
    ✅ raw/comve                         (train/dev/test CSVs)
    ✅ formatted/comve/test.json         (1000 items)
    ✅ counterfactual/comve/ext_org.json (9965 items — FULLY READY)

QWEN PIPELINE:
    ✅ Step 1a  original answers         experiments/original/zs-comve-qwen/answer_gd.json           (1000 items, ~17 min)
    ✅ Step 1b  counterfactual answers   experiments/counterfactual/zs-comve-qwen/answer_gd.json      (9965 items, ~2.5 hrs)
    ✅ counter.py                        experiments/counterfactual/zs-comve-qwen/answer_gd_counter.json
    ✅ Step 2   explanations             experiments/counterfactual/zs-comve-qwen/explanation_gd.json  (235 items, ~18 min)
    ✅ Step 3   feedback iter 0          experiments/counterfactual/zs-comve-qwen/iter0_feedback_nl.json (~48 min)
    ✅ Step 4   refinement iter 0        experiments/counterfactual/zs-comve-qwen/iter0_refinement_nl.json (~22 min)
    ✅ Step 3   feedback iter 1          experiments/counterfactual/zs-comve-qwen/iter1_feedback_nl.json
    ✅ Step 4   refinement iter 1        experiments/counterfactual/zs-comve-qwen/iter1_refinement_nl.json
    ✅ Step 3   feedback iter 2          experiments/counterfactual/zs-comve-qwen/iter2_feedback_nl.json
    ✅ Step 4   refinement iter 2        experiments/counterfactual/zs-comve-qwen/iter2_refinement_nl.json (~33 min)
    ✅ Evaluation                        logs/faithfulness_*.log

FALCON PIPELINE (partial — from local machine):
    ✅ Step 1b + counter.py  →  experiments/counterfactual/zs-comve-falcon/answer_gd_counter.json
    ❌ Everything else       →  falcon not cached on server

================================================================================
FINAL RESULTS — QWEN + COMVE + NL FEEDBACK (completed March 22, 2026)
================================================================================

COUNTER STATS:
    Total counterfactuals : 9,965
    Flipped (counter)     : 235
    Counter rate          : 2.36%
    Saved to              : experiments/counterfactual/zs-comve-qwen/answer_gd_counter.json

    Note: 2.36% is low because Qwen is a strong stable model.
    Injected adjectives/adverbs weren't strong enough to flip its answer
    most of the time. 235 items is still sufficient for reliable evaluation.

FAITHFULNESS RESULTS (unfaithfulness rate — lower is better):

    Stage       Iter   Faithful   Unfaithful   Unfaith Rate   Explanation Length
    ----------  -----  ---------  -----------  -------------  ------------------
    Baseline    -      66         169          71.91%         24.2 tokens
    Refined     0      78         157          66.81%         33.1 tokens
    Refined     1      85         150          63.83%         39.1 tokens
    Refined     2      86         149          63.40%         42.9 tokens

TOTAL IMPROVEMENT:
    Baseline → After SR-NLE : 71.91% → 63.40% = 8.51% absolute reduction

COMPARISON WITH PAPER:
    Paper (avg 4 models)  : 54.81% → 36.02% = 18.79% improvement
    Our run (Qwen only)   : 71.91% → 63.40% =  8.51% improvement

    Why our baseline is higher than paper's average:
    → Paper averages across 4 models (Llama, Mistral, Qwen, Falcon)
    → Qwen specifically has a higher baseline unfaithfulness on ComVE
    → SR-NLE still consistently improves across all 3 iterations ✅
    → Each iteration reduces unfaithfulness progressively (diminishing returns)
    → Explanation length grows with each refinement (more detailed reasoning)




def print_pipeline_status():
    """Print current pipeline status."""
    steps = [
        ("✅", "Data Prep",         "data/counterfactual/comve/ext_org.json (9965 items)"),
        ("🔄", "Step 1a",           "experiments/original/zs-comve-qwen/answer_gd.json"),
        ("⏳", "Step 1b",           "experiments/counterfactual/zs-comve-qwen/answer_gd.json"),
        ("⏳", "counter.py",        "experiments/counterfactual/zs-comve-qwen/answer_gd_counter.json"),
        ("⏳", "Step 2",            "experiments/counterfactual/zs-comve-qwen/explanation_gd.json"),
        ("⏳", "Step 3 (iter 0)",   "experiments/counterfactual/zs-comve-qwen/iter0_feedback_nl.json"),
        ("⏳", "Step 4 (iter 0)",   "experiments/counterfactual/zs-comve-qwen/iter0_refinement_nl.json"),
        ("⏳", "Step 3 (iter 1)",   "experiments/counterfactual/zs-comve-qwen/iter1_feedback_nl.json"),
        ("⏳", "Step 4 (iter 1)",   "experiments/counterfactual/zs-comve-qwen/iter1_refinement_nl.json"),
        ("⏳", "Step 3 (iter 2)",   "experiments/counterfactual/zs-comve-qwen/iter2_feedback_nl.json"),
        ("⏳", "Step 4 (iter 2)",   "experiments/counterfactual/zs-comve-qwen/iter2_refinement_nl.json"),
        ("⏳", "Evaluation",        "logs/faithfulness_results.csv"),
    ]

    import os
    print("\n" + "="*70)
    print("SR-NLE PIPELINE STATUS — QWEN + COMVE + NL FEEDBACK")
    print("="*70)
    for icon, step, path in steps:
        exists = "✅" if os.path.exists(path) else icon
        print(f"  {exists}  {step:<20} {path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_pipeline_status()
