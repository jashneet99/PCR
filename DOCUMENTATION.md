# SR-NLE + PCR: Detailed Project Documentation

**Project:** Improving Faithfulness of Natural Language Explanations via Self-Refinement and Probabilistic Counterfactual Refinement  
**Dataset:** ComVE (Commonsense Validation and Explanation)  
**Models:** Qwen2.5-7B-Instruct, Falcon3-7B-Instruct (8-bit quantized)  
**Last Updated:** 2026-04-13

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset: ComVE](#2-dataset-comve)
3. [SR-NLE: Self-critique and Refinement for Natural Language Explanations](#3-sr-nle-self-critique-and-refinement-for-natural-language-explanations)
   - 3.1 [Motivation](#31-motivation)
   - 3.2 [Counterfactual Data Generation](#32-counterfactual-data-generation)
   - 3.3 [Counter Rate (Counterfactual Sensitivity)](#33-counter-rate-counterfactual-sensitivity)
   - 3.4 [Pipeline](#34-pipeline)
   - 3.5 [Feedback Mechanisms](#35-feedback-mechanisms)
   - 3.6 [Faithfulness Metric](#36-faithfulness-metric)
   - 3.7 [SR-NLE Results](#37-sr-nle-results)
   - 3.8 [Limitations of SR-NLE](#38-limitations-of-sr-nle)
4. [PCR: Probabilistic Counterfactual Refinement](#4-pcr-probabilistic-counterfactual-refinement)
   - 4.1 [Motivation: Why PCR was Needed](#41-motivation-why-pcr-was-needed)
   - 4.2 [Core Objects: DE, CE, ME](#42-core-objects-de-ce-me)
   - 4.3 [PCR Phase 1: Prompt-Only Mutual Exclusivity](#43-pcr-phase-1-prompt-only-mutual-exclusivity)
   - 4.4 [PCR Phase 2: Probabilistic Margin Guidance](#44-pcr-phase-2-probabilistic-margin-guidance)
   - 4.5 [PCR Results](#45-pcr-results)
5. [Full Results Comparison](#5-full-results-comparison)
6. [Key Findings and Insights](#6-key-findings-and-insights)
7. [Project Structure](#7-project-structure)
8. [How to Run](#8-how-to-run)

---

## 1. Problem Statement

Large language models (LLMs) generate natural language explanations (NLEs) to justify their predictions. These explanations are called **post-hoc NLEs** — they are generated after the answer, not used to compute it.

The core problem: **post-hoc NLEs are often unfaithful** — they do not actually reflect the model's internal reasoning. A model might predict answer A for one reason, but its explanation could equally describe answer B. The explanation is generic and vague, not causally tied to the prediction.

**Faithfulness** is measured by a proxy: after injecting a word edit into the input that flips the model's answer, does the model's explanation mention that specific edit word? If yes → faithful. If no → unfaithful.

This project explores two approaches to improve faithfulness:
- **SR-NLE** — iterative self-critique and refinement (baseline from EMNLP 2025 paper)
- **PCR** — our extension that uses counterfactual contrastive reasoning

---

## 2. Dataset: ComVE

**Task:** Given two sentences, identify which one violates commonsense.

```
Sentence 0: "A grizzly fierce bear likes honey"   ← violates commonsense (edit: "fierce")
Sentence 1: "A grizzly bear likes honey"
Answer: A (Sentence 0 is wrong)
```

**Answer options:** (A) = Sentence 0, (B) = Sentence 1

**Test set:** 1,000 items  
**Counterfactual set:** ~9,965 items (multiple word-edited variants per original item)  
**Counterfactual generation:** An LLM generates word substitutions in the "wrong" sentence that change its meaning. These edits are extracted and applied to create counterfactual test items.

The `edit_word` is the specific word injected via the counterfactual edit. It acts as the faithfulness signal: if the explanation mentions `edit_word`, it is considered faithful.

---

## 3. SR-NLE: Self-critique and Refinement for Natural Language Explanations

**Paper:** [arXiv:2505.22823](https://arxiv.org/abs/2505.22823)  
**Conference:** EMNLP 2025  
**Authors:** Yingming Wang, Pepa Atanasova

### 3.1 Motivation

SR-NLE is built on the hypothesis that LLMs can improve their own explanations through iterative self-feedback — without any external supervision, additional training, or fine-tuning. The key insight is that feedback mechanisms can guide the model to produce explanations that are more causally tied to their predictions.

### 3.2 Counterfactual Data Generation

The SR-NLE faithfulness test requires counterfactual items — inputs where a single word edit flips the model's predicted answer.

**Step 1 — Generate edits** ([src/data_gen/generate_edits.py](SR-NLE/src/data_gen/generate_edits.py))

An LLM (e.g., Llama-3.1-8B) is prompted with the "wrong" sentence and asked to generate multiple word substitution edits. For ComVE, the prompt uses 10 in-context examples.

```
Input:  "A grizzly bear likes honey"
Output: ["fierce bear", "cruel bear", "feral bear", "tame bear", ...]
```

The edited sentences are stored in `data/counterfactual/comve/`.

**Step 2 — Format** ([src/data_format/format_dataset.py](SR-NLE/src/data_format/format_dataset.py))

Raw data is formatted into a unified JSON structure in `data/formatted/`.

### 3.3 Counter Rate (Counterfactual Sensitivity)

Not all edits actually flip the model's answer. The **counter rate** measures what percentage of counterfactual items cause the model to change its prediction.

**How it works** ([src/evaluation/counter.py](SR-NLE/src/evaluation/counter.py)):

```
For each counterfactual item:
    org_pred = model answer on original sentence
    ct_pred  = model answer on counterfactually edited sentence
    If org_pred != ct_pred → "counter" item (flipped)

Counter Rate = (# flipped) / (# valid)
```

Only **flipped items** are used for the explanation + SR-NLE + PCR pipeline. This ensures we are testing explanations on inputs where the answer itself was sensitive to the edit — making faithfulness meaningful.

| Model  | Flipped Items | Total  | Counter Rate |
|--------|--------------|--------|--------------|
| Qwen   | 235          | 9,965  | 2.36%        |
| Falcon | 178          | 9,965  | 1.79%        |

Qwen has a slightly higher counter rate, indicating it is more sensitive to word-level edits in the input.

### 3.4 Pipeline

The full SR-NLE pipeline runs in sequence. All runners are in [src/runners/](SR-NLE/src/runners/).

```
Step 1a  →  answer_runner.py  (original)       → answer on 1,000 original items
Step 1b  →  answer_runner.py  (counterfactual) → answer on ~9,965 counterfactual items
Counter  →  counter.py                          → filter: keep only flipped items
Step 2   →  explanation_runner.py              → generate initial NLE for flipped items
Step 3   →  feedback_runner.py                 → generate feedback on explanation
Step 4   →  refinement_runner.py               → refine explanation using feedback
          (Steps 3-4 repeat: iteration 0, 1, 2)
Eval     →  faithfulness.py                    → measure faith rate
```

**Step 1 — Answer Generation** ([src/runners/answer_runner.py](SR-NLE/src/runners/answer_runner.py))

The model is prompted in zero-shot mode to identify which sentence violates commonsense, without any explanation. The prompt format:

```
You are given two sentences. Identify which one violates commonsense.
Sentence 0: {sentence0}
Sentence 1: {sentence1}
Answer Options: [OPTIONS]
Please select the most appropriate answer without any explanation.
You must give your answer only in the following format: Answer: (X)
```

**Step 2 — Explanation Generation** ([src/runners/explanation_runner.py](SR-NLE/src/runners/explanation_runner.py))

For each flipped item, the model is asked to explain its answer (the answer it gave on the counterfactual input). This produces the **initial explanation** (`explanation_gd.json`).

```
You are given two sentences, and you have selected the one that violates commonsense.
...
Your selected answer is: (A).
Now, please provide an explanation for your choice.
Explanation: [your explanation here.]
```

**Steps 3–4 — Feedback + Refinement (Iterative)**

For each iteration (0, 1, 2):

- **Feedback runner** reads the current explanation, generates feedback
- **Refinement runner** reads the explanation + feedback, produces a refined explanation
- The refined explanation becomes input for the next iteration's feedback

Output files per iteration:
```
iter0_feedback_{type}.json
iter0_refinement_{type}.json
iter1_feedback_{type}.json
iter1_refinement_{type}.json
iter2_feedback_{type}.json
iter2_refinement_{type}.json
```

### 3.5 Feedback Mechanisms

SR-NLE supports four feedback types, each providing a different signal to guide refinement:

#### (a) NL — Natural Language Self-Feedback

The model critiques its own explanation in natural language.

**Feedback prompt:**
```
Your selected answer is: (A)
Your explanation is: [EXPLANATION]

Now, please provide feedback on this explanation.
- Identify whether the explanation accurately reflects your actual reasoning.
- Point out missing, unclear, or incorrect details.
- Describe what should be added or revised.
- State that no improvement is needed if the explanation is good enough.
Feedback: [your feedback here.]
```

**Refinement prompt:**
```
Your explanation is: [EXPLANATION]
The feedback you received is: [FEEDBACK]
If no improvement is needed, repeat the original explanation.
Otherwise, refine based on feedback.
Refined Explanation: [your refined explanation here.]
```

#### (b) IW — Important Words (LLM-ranked)

The model ranks input words by importance (score 1–100). The top-ranked words are passed as feedback signal.

```
Evaluate all words in Sentence 0 and Sentence 1.
Rank them by importance to your decision.
Format: <rank>. <word>, <score>
```

Refinement: the model is told to ensure its explanation naturally integrates the important words.

#### (c) AIW-ATTN — Attention-based Attribution

Attention weights from the model's final layer are used to score input tokens by their influence on the answer prediction. Words with highest aggregated attention scores are passed as feedback.

Implemented in [src/attribution/attention.py](SR-NLE/src/attribution/attention.py) using `AttentionAttribution`. Aggregation: last-layer abs_mean over answer tokens, then per-word sum.

#### (d) AIW-IG — Integrated Gradients Attribution

Integrated Gradients (IG) computes input token importance by integrating gradients from a baseline (zero embedding) to the actual input. More computationally expensive than attention but more theoretically grounded.

Implemented in [src/attribution/integrated_gradient.py](SR-NLE/src/attribution/integrated_gradient.py).

### 3.6 Faithfulness Metric

**Definition** ([src/evaluation/faithfulness.py](SR-NLE/src/evaluation/faithfulness.py)):

```python
def is_word_in_expl(word, expl):
    return word.lower() in expl.lower()
```

For each flipped item:
- `edit_word` = the word injected to flip the model's answer
- `explanation.final` = the generated NLE

**Faith Rate = (# items where `edit_word` in explanation) / (# total valid items)**

This is a necessary-condition proxy for faithfulness. An explanation that mentions the `edit_word` is more likely to be causally explaining the counterfactual difference that actually drove the answer change.

### 3.7 SR-NLE Results

#### Qwen — ComVE (235 flipped items)

| Stage              | Faithful | Total | Faith Rate | Improvement vs Init |
|--------------------|----------|-------|------------|---------------------|
| Init (explanation) | 66       | 235   | 28.09%     | —                   |
| Iter 0 refinement  | 78       | 235   | 33.19%     | +5.10%              |
| Iter 1 refinement  | 85       | 235   | 36.17%     | +8.08%              |
| Iter 2 refinement  | 86       | 235   | 36.60%     | +8.51%              |

**SR-NLE best (Qwen): 28.09% → 36.60% (+8.51% absolute)**

Improvement is significant initially but plateaus sharply:
- Iter 0→1: +2.98%
- Iter 1→2: +0.43% ← signal nearly exhausted

#### Falcon — ComVE (178 flipped items)

| Stage              | Faithful | Total | Faith Rate | Improvement vs Init |
|--------------------|----------|-------|------------|---------------------|
| Init (explanation) | 66       | 178   | 37.08%     | —                   |
| Iter 0 refinement  | 68       | 178   | 38.20%     | +1.12%              |
| Iter 1 refinement  | 66       | 178   | 37.08%     | +0.00%              |
| Iter 2 refinement  | 75       | 178   | 42.13%     | +5.05%              |

**SR-NLE best (Falcon): 37.08% → 42.13% (+5.05% absolute)**

Falcon starts at a higher baseline (37.08% vs 28.09%) but shows non-monotonic improvement — iter 1 is identical to init, and the big gain only comes at iter 2.

### 3.8 Limitations of SR-NLE

SR-NLE's feedback signal is **shallow**: it asks the model to mention the `edit_word`. A model can satisfy this by mechanically inserting the word into an otherwise vague explanation. Example:

```
Sentence 0: "A grizzly fierce bear likes honey"
Sentence 1: "A grizzly bear likes honey"

SR-NLE explanation: "Grizzly bears are known to love honey,
                      so Sentence 0 violates commonsense."

SR-NLE says: UNFAITHFUL — "fierce" not mentioned.
SR-NLE fix:  "Grizzly fierce bears are known to love honey,
               so Sentence 0 violates commonsense."

SR-NLE says: FAITHFUL — "fierce" mentioned.
```

The word was inserted but the explanation is still logically vague — it could describe either sentence. The explanation does not actually prove WHY answer A is correct rather than B.

**The signal saturates**: After 2-3 iterations, there are no more easy wins. Items that did not become faithful after iter 2 are ones where the model could not generate better feedback, because the feedback prompt itself only asks "what is wrong?" without providing a specific direction.

---

## 4. PCR: Probabilistic Counterfactual Refinement

PCR is our extension built on top of SR-NLE. It addresses the shallow feedback problem through a fundamentally different refinement objective: **contrastive mutual exclusivity**.

### 4.1 Motivation: Why PCR was Needed

SR-NLE's question to the model: *"Did you mention the edit word?"*

PCR's question: *"Does your explanation EXCLUSIVELY prove your answer, such that the opposite answer cannot also be justified from it?"*

This is stronger. An explanation is truly faithful not just because it mentions the right word, but because:
1. Reading it alone, you can confidently identify the answer as y
2. Reading the counterfactual explanation alone, you can confidently identify the answer as y_prime
3. The two explanations do not overlap or contradict each other

PCR operationalizes this through **counterfactual explanations (CE)** — explanations generated from the opposite perspective — and uses these as a contrastive reference to sharpen the draft explanation (DE).

The word inclusion becomes a **side effect** of logical specificity, not the goal.

### 4.2 Core Objects: DE, CE, ME

| Symbol | Name | Description |
|--------|------|-------------|
| `x` | Input | Sentence 0 + Sentence 1 |
| `y` | Factual Answer | Model's actual prediction (e.g., "A") |
| `y'` | Counterfactual Answer | The opposite answer (e.g., "B") |
| `DE` | Draft Explanation | Explanation for y — starts as SR-NLE's output, gets refined |
| `CE` | Counterfactual Explanation | Explanation for y_prime — generated by PCR, used as contrastive reference |
| `ME` | Mutual Exclusivity | Phase 1 signal: YES if DE and CE are logically exclusive, NO if they overlap |
| `Δt` | Margin for DE | Phase 2 signal: P(y\|DE) − P(y'\|DE) — how strongly DE predicts y |
| `Δ't` | Margin for CE | Phase 2 signal: P(y'\|CE) − P(y\|CE) — how strongly CE predicts y' |
| `τ` | Threshold | Phase 2 convergence criterion: both margins must reach 0.95 |

### 4.3 PCR Phase 1: Prompt-Only Mutual Exclusivity

**Runner:** [PCR/src/runners/pcr_phase1_runner.py](SR-NLE/PCR/src/runners/pcr_phase1_runner.py)  
**Input:** `experiments/counterfactual/zs-comve-{model}/explanation_gd.json` (SR-NLE baseline)  
**Output:** `PCR/experiments/counterfactual/zs-comve-{model}/pcr_phase1.json`

#### Algorithm

```
For each item:
  Step 1: DE = SR-NLE explanation for y
          Generate CE = explanation for y_prime (from opposite perspective)

  Loop (up to max_iters=3):
    Step 2: ME check — "Are DE and CE mutually exclusive?" → YES/NO
            If YES → converged = True, stop
            If NO  → continue

    Step 3: Critique + Refine DE
            Prompt: "These are NOT mutually exclusive. Rewrite DE so it:
                     1. Clearly proves why (y) is the answer
                     2. Explicitly excludes the reasoning of (y_prime)"

    Step 4: Critique + Refine CE (roles swapped)
            Same prompt but from y_prime's perspective

  Final: de_final = best DE after all iterations
```

#### Prompts Used

**CE Generation** ([PCR/src/prompts/comve_prompts.py](SR-NLE/PCR/src/prompts/comve_prompts.py)):
```
The actual answer is (y) — that sentence violates commonsense.
Now imagine the answer were (y_prime) instead.
Explain in one concise sentence why (y_prime) might violate commonsense.
Explanation: [your explanation here.]
```

**Mutual Exclusivity Check:**
```
Explanation for answer (y):        DE
Explanation for answer (y_prime):  CE

Are these two explanations mutually exclusive?
Mutually exclusive means: reading Explanation 1 alone, you can clearly tell
the answer is (y) and NOT (y_prime), and vice versa.
Answer with only YES or NO.
```

**Critique (when ME=NO):**
```
The explanations are overlapping.
Rewrite the explanation for (y) so it:
  1. Clearly proves why (y) is the answer
  2. Explicitly excludes the reasoning of (y_prime)
Explanation: [your explanation here.]
```

#### Worked Examples

**Example A — Converged immediately (iterations = 0)**
```
Sentence 0: "Grizzly cruel bears hate honey."
Sentence 1: "Grizzly bears love honey."
edit_word: "cruel"

DE: "Grizzly bears are attracted to honey's sweet aroma, so Sentence 1 is correct."
CE: "Grizzly bears hating honey contradicts their known attraction to sweet smells."

ME check → YES (DE points to Sentence 1, CE points to Sentence 0 — mutually exclusive)
Result: converged, no refinement needed, iterations = 0
Note: "cruel" still not in DE → unfaithful but converged early
```

**Example B — Converged after 1 iteration (iterations = 1)**
```
Sentence 0: "Grizzly fierce bears hate honey."
Sentence 1: "Grizzly bears love honey."
edit_word: "fierce"

iter -1:
  DE: "Grizzly bears attracted to honey's scent, Sentence 0 violates CS."
  CE: "Bears hating honey contradicts common experience."
  ME: NO ← both vague, cannot distinguish which sentence is wrong

Critique DE → iter 0:
  DE: "Sentence 0 states grizzly FIERCE bears HATE honey — directly
       contradicts known behavior. Explanation B focuses on general
       attraction, but Sentence 0 specifically claims they hate honey."
  ME: YES ← specific to Sentence 0 now

Result: "fierce" in de_final → FAITHFUL. Iterations = 1.
```

**Example C — Never converged (iterations = 3)**
```
Sentence 0: "Grizzly feral bears hate honey."
Sentence 1: "Grizzly bears love honey."
edit_word: "feral"

All 3 iterations: ME = NO
(Model kept writing CE that argued Sentence 0 is wrong — confused about CE's role)

BUT: de_final contains "feral bears hate honey" → FAITHFUL
Conclusion: convergence and faithfulness are NOT the same thing.
```

#### Iteration Distribution (Qwen, 235 items)

| Iterations | Items | Description |
|-----------|-------|-------------|
| 0 | 98 | Converged at first ME check — no refinement needed |
| 1 | 28 | Converged after 1 critique round |
| 2 | 9 | Converged after 2 critique rounds |
| 3 | 100 | Hit max_iters, did not formally converge |

Total converged: 135/235 = 57.4%  
Average iterations: 1.47

#### Falcon Phase 1 Behavior

| Iterations | Items | Description |
|-----------|-------|-------------|
| 0 | 166 | Converged immediately |
| 1-3 | 12 | Needed refinement |

Falcon converges much faster (166/178 = 93.3% in 0 iterations, avg 0.49 iters). This means Falcon's initial explanations already appear mutually exclusive to the model — but this does NOT translate to a big faithfulness improvement. Falcon Phase 1 only achieved 38.76% faith rate vs 37.08% baseline (marginal gain), while Qwen Phase 1 jumped from 28.09% to 53.62%.

**Insight:** High ME convergence does not guarantee faithfulness. Falcon's model says "YES, these are mutually exclusive" too readily — a subjective judgment that may not be well-calibrated. This weakness motivates Phase 2.

### 4.4 PCR Phase 2: Probabilistic Margin Guidance

**Runner:** [PCR/src/runners/pcr_phase2_runner.py](SR-NLE/PCR/src/runners/pcr_phase2_runner.py)  
**Input:** `experiments/counterfactual/zs-comve-{model}/explanation_gd.json` (same as Phase 1, NOT Phase 1 output)  
**Output:** `PCR/experiments/counterfactual/zs-comve-{model}/pcr_phase2.json`

Phase 2 replaces the subjective YES/NO mutual exclusivity question with an **objective mathematical signal**. Instead of asking the model whether its explanation is faithful, Phase 2 **measures** it using the model's own probability distribution.

#### Core Innovation: `get_margin()`

Implemented in [PCR/src/model/model.py](SR-NLE/PCR/src/model/model.py):

```python
def get_margin(self, explanation: str, y: str, y_prime: str) -> float:
    # Build a prompt with ONLY the explanation (no original sentences x)
    prompt = f"Based only on the following explanation, predict the answer.\n\n"
             f"Explanation: {explanation}\n\nAnswer (choose one):"

    # Tokenize and get logits at the last token position
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model(**inputs)
    last_logits = output.logits[0, -1, :]

    # Extract logits for y and y_prime tokens
    y_token_id       = get_answer_token_id(y)
    y_prime_token_id = get_answer_token_id(y_prime)

    # Binary softmax over the two relevant logits
    probs = F.softmax([last_logits[y_token_id], last_logits[y_prime_token_id]])

    # Compute margin Δt
    delta_t = P(y|explanation) - P(y_prime|explanation)
    return delta_t
```

**Interpretation of Δt:**
| Δt Value | Meaning |
|----------|---------|
| +1.0 | Explanation perfectly and exclusively predicts y (fully faithful) |
| 0.0 | Explanation is ambiguous (equal probability for y and y') |
| -1.0 | Explanation pushes toward the WRONG answer |

This is computed **without showing the model the original sentences** — only the explanation. This directly measures whether the explanation alone is sufficient to reproduce the model's answer.

#### Algorithm

```
For each item:
  DE = SR-NLE explanation, CE = generated counterfactual explanation

  Loop (up to max_iters=3):
    Step 1: Δt  = get_margin(DE, y, y_prime)   → how strongly DE predicts y
            Δ't = get_margin(CE, y_prime, y)    → how strongly CE predicts y_prime

    Step 2: Convergence check
            If Δt >= τ (0.95) AND Δ't >= τ → converged, stop

    Step 3: Compute GAPs
            GAP_DE = τ - Δt    (how far DE is from threshold)
            GAP_CE = τ - Δ't   (how far CE is from threshold)

    Step 4: GAP-informed critique
            Tell model exactly: "Your explanation has margin = Δt
                                  (needs GAP_DE more to reach threshold 0.95)"
            → Refine DE and CE

  Final: de_final = last DE after max_iters
```

#### Phase 2 Prompts

**GAP-Informed Critique for DE:**
```
Faithfulness measurement results:
  - Explanation for (y):    margin = 0.40  (needs 0.55 more to reach threshold 0.95)
  - Explanation for (y'):   margin = 0.35  (needs 0.60 more to reach threshold 0.95)

Both margins are below the required threshold of 0.95.
This means the explanation for (y) is still ambiguous.

Rewrite the explanation for (y) so it more specifically and precisely proves
why (y) violates commonsense and explicitly excludes the reasoning of (y').
Explanation: [your explanation here.]
```

The key difference from Phase 1: the model receives **a number** telling it exactly how far the explanation is from being faithful. This is a calibrated, objective signal rather than a subjective opinion.

### 4.5 PCR Results

#### Qwen — ComVE (235 items)

| Stage           | Faithful | Total | Faith Rate | vs SR-NLE Init | vs SR-NLE Best | Converged  | Avg Iters |
|-----------------|----------|-------|------------|----------------|----------------|------------|-----------|
| SR-NLE Init     | 66       | 235   | 28.09%     | —              | —              | —          | —         |
| SR-NLE Iter 2   | 86       | 235   | 36.60%     | +8.51%         | —              | —          | —         |
| PCR Phase 1     | 126      | 235   | 53.62%     | +25.53%        | +17.02%        | 135/235    | 1.47      |
| PCR Phase 2     | 154      | 235   | 65.53%     | +37.44%        | +28.93%        | 0/235      | 4.00      |

**PCR Phase 1 vs SR-NLE best: +17.02%**  
**PCR Phase 2 vs SR-NLE best: +28.93%**  
**PCR Phase 2 vs PCR Phase 1: +11.91% additional improvement**

Phase 2 average final margins:
- Average Δt  = 0.1415 (DE still ambiguous in absolute terms)
- Average Δ't = −0.1334 (CE consistently negative — model cannot generate faithful CEs)

#### Falcon — ComVE (178 items)

| Stage           | Faithful | Total | Faith Rate | vs SR-NLE Init | vs SR-NLE Best | Converged  | Avg Iters |
|-----------------|----------|-------|------------|----------------|----------------|------------|-----------|
| SR-NLE Init     | 66       | 178   | 37.08%     | —              | —              | —          | —         |
| SR-NLE Iter 2   | 75       | 178   | 42.13%     | +5.05%         | —              | —          | —         |
| PCR Phase 1     | 69       | 178   | 38.76%     | +1.68%         | −3.37%         | 166/178    | 0.49      |
| PCR Phase 2     | 93       | 178   | 52.25%     | +15.17%        | +10.12%        | 0/178      | 5.00      |

**Notable:** Falcon Phase 1 is actually slightly worse than SR-NLE best (42.13% → 38.76%). Because 166/178 items converged at iteration 0 (ME=YES immediately), Phase 1 performed almost no refinement. The high convergence rate masked poor performance — the model was too easily satisfied with its own mutual exclusivity judgment.

**Phase 2 rescued Falcon**: Despite Phase 1 underperforming, Phase 2 pushed Falcon from 42.13% to 52.25% (+10.12%) using objective margin guidance.

---

## 5. Full Results Comparison

### Complete Table (Both Models, All Stages)

| Stage               | Qwen Faith Rate | Qwen Faithful | Falcon Faith Rate | Falcon Faithful |
|---------------------|-----------------|---------------|-------------------|-----------------|
| SR-NLE Init         | 28.09%          | 66/235        | 37.08%            | 66/178          |
| SR-NLE Iter 0       | 33.19%          | 78/235        | 38.20%            | 68/178          |
| SR-NLE Iter 1       | 36.17%          | 85/235        | 37.08%            | 66/178          |
| SR-NLE Iter 2       | 36.60%          | 86/235        | 42.13%            | 75/178          |
| PCR Phase 1         | 53.62%          | 126/235       | 38.76%            | 69/178          |
| PCR Phase 2         | 65.53%          | 154/235       | 52.25%            | 93/178          |

### Unfaithfulness Reduction (Qwen)

```
SR-NLE init → PCR Phase 2:
   Unfaithful: 169 → 81 items
   Reduction: 71.91% → 34.47% unfaithfulness
   Absolute reduction: −37.44 percentage points
```

---

## 6. Key Findings and Insights

### Finding 1: SR-NLE Helps But Plateaus Quickly

SR-NLE provides consistent initial improvement (especially for Qwen: +8.51%), but the gains diminish rapidly after iteration 1. The NL feedback signal exhausts itself — after 2 iterations, the model cannot generate meaningfully different feedback on the same explanation.

### Finding 2: PCR's Contrastive Signal Breaks the Plateau

By introducing a counterfactual explanation (CE) as a reference, PCR forces the model to think about what makes its answer distinct from the alternative. This contrastive objective naturally leads to more specific explanations that include the edit_word as a side effect — not as a goal.

SR-NLE: word inclusion is the **goal**  
PCR: word inclusion is a **consequence** of logical specificity

### Finding 3: Convergence ≠ Faithfulness

In Phase 1, an item can achieve ME convergence (model says "YES, mutually exclusive") without the `edit_word` appearing in the final explanation. Conversely, items that never formally converged can still be faithful (the DE was improved enough to mention the edit_word even if the CE problem remained unresolved).

Phase 1 convergence rate: 135/235 = 57.4%  
Phase 1 faith rate: 126/235 = 53.62%  
These are different things.

### Finding 4: Phase 2 Formal Convergence = 0, Yet Best Results

Phase 2 never converged (τ=0.95 is a very strict threshold). Average Δt only reached 0.1415 — far below 0.95. Yet Phase 2 achieved the highest faith rate (65.53%). This demonstrates that:

- The **GAP-informed critique** (telling the model "you need 0.55 more margin") produces better refinements than the YES/NO judgment, even without formal convergence
- The probability signal is useful as **directional guidance** even when the threshold is unreachable

### Finding 5: CE Margin Is Persistently Negative (Δ't = −0.1334)

In Phase 2, the average final Δ't for CE is negative (-0.1334). This means the CE consistently pushes the model toward the wrong answer — the counterfactual explanation, when fed back to the model alone, causes it to predict y (the original answer) rather than y' (what the CE is supposed to support).

This is a systematic finding: **the model cannot generate faithful counterfactual explanations**. It is resistant to explaining the opposite scenario from the one it actually predicted. This is an intrinsic limitation of the model's world model and causal reasoning under hypothetical framing.

### Finding 6: Falcon vs Qwen — Different Failure Modes

- **Qwen**: Lower initial faithfulness (28.09%), but benefits greatly from PCR (+37.44% over init)
- **Falcon**: Higher initial faithfulness (37.08%), converges too quickly in Phase 1 (93.3% in 0 iters), benefits mostly from Phase 2's objective signal (+15.17% over init via Phase 2)

Falcon's faster convergence in Phase 1 suggests its self-evaluation (ME judgment) is less discriminative — it accepts vague explanations as "mutually exclusive." Phase 2's mathematical criterion bypasses this miscalibration.

### Finding 7: PCR Has Zero Regressions in Phase 1 (Qwen)

No item that was faithful in SR-NLE baseline became unfaithful after PCR Phase 1. The refinement only moves items from unfaithful to faithful, never the reverse.

---

## 7. Project Structure

```
SR-NLE/
├── README.md                           ← Original SR-NLE paper README
├── RESULTS.md                          ← High-level results summary
├── DOCUMENTATION.md                    ← This file
├── falcon_results.md                   ← Falcon-specific notes
├── requirements.txt
├── pipeline_guide.py                   ← Step-by-step pipeline guide
├── results_presentation.py             ← SR-NLE results display
│
├── configs/                            ← Hydra configs for each runner
│   ├── answer.yaml
│   ├── explanation.yaml
│   ├── feedback.yaml
│   └── refinement.yaml
│
├── src/
│   ├── data_format/
│   │   └── format_dataset.py           ← Raw → unified JSON format
│   ├── data_gen/
│   │   ├── generate_edits.py           ← LLM-based word edit generation
│   │   └── edit_prompt_10.txt          ← 10-shot prompt for edits
│   ├── evaluation/
│   │   ├── counter.py                  ← Counterfactual sensitivity (counter rate)
│   │   └── faithfulness.py             ← edit_word in explanation metric
│   ├── model/
│   │   └── model.py                    ← GenerationModel (8-bit, all 4 LLMs)
│   ├── modules/
│   │   ├── answer_generator.py
│   │   ├── explanation_generator.py
│   │   ├── feedback_generator.py       ← NL / IW / AIW feedback
│   │   └── refinement_generator.py
│   ├── runners/
│   │   ├── answer_runner.py
│   │   ├── explanation_runner.py
│   │   ├── feedback_runner.py
│   │   └── refinement_runner.py
│   ├── attribution/
│   │   ├── attention.py                ← Attention-based IW attribution
│   │   └── integrated_gradient.py     ← IG-based IW attribution
│   └── prompts/
│       └── zero_shot/
│           └── comve_prompt.py         ← All SR-NLE prompt templates
│
├── data/
│   ├── formatted/comve/test.json       ← 1,000 original ComVE items
│   └── counterfactual/comve/           ← ~9,965 counterfactual items
│
├── experiments/
│   └── counterfactual/zs-comve-{model}/
│       ├── answer_gd.json              ← Step 1a output
│       ├── answer_gd_counter.json      ← Flipped items only
│       ├── explanation_gd.json         ← Step 2 output (SR-NLE baseline)
│       ├── iter0_feedback_nl.json
│       ├── iter0_refinement_nl.json
│       ├── iter1_feedback_nl.json
│       ├── iter1_refinement_nl.json
│       ├── iter2_feedback_nl.json
│       └── iter2_refinement_nl.json
│
├── results/ComVE/
│   ├── ME.CSV                          ← Full comparison table (both models)
│   ├── all_results.txt
│   ├── comparison_table.csv
│   ├── qwen_results.csv
│   └── falcon_results.csv
│
└── PCR/
    ├── pcr_phase1_guide.py             ← Detailed Phase 1 documentation
    ├── pcr_phase1_results_presentation.py
    ├── pcr_final_results.py            ← Complete results summary
    ├── src/
    │   ├── model/
    │   │   └── model.py                ← Extended model with get_margin()
    │   ├── prompts/
    │   │   └── comve_prompts.py        ← All PCR prompt templates
    │   ├── runners/
    │   │   ├── pcr_phase1_runner.py    ← Phase 1 ME-based loop
    │   │   └── pcr_phase2_runner.py    ← Phase 2 probability-margin loop
    │   └── evaluation/
    │       └── faithfulness.py         ← PCR-specific faith evaluation
    ├── experiments/counterfactual/zs-comve-{model}/
    │   ├── pcr_phase1.json
    │   └── pcr_phase2.json
    └── logs/df/
        ├── pcr_phase1_faithfulness.csv
        ├── pcr_faithfulness_comparison.csv
        └── pcr_final_results_summary.csv
```

---

## 8. How to Run

### Environment Setup

```bash
cd /home/dibyanayan/jashneet/SR-NLE
source /home/dibyanayan/jashneet/sr-nle-test/bin/activate
```

### SR-NLE Full Pipeline

```bash
# Step 1a: Answer on original items
python src/runners/answer_runner.py \
    --config configs/answer.yaml \
    dataset.type=original prompt.type=zs dataset.name=comve \
    model.name=qwen decoding.type=gd

# Step 1b: Answer on counterfactual items
python src/runners/answer_runner.py \
    --config configs/answer.yaml \
    dataset.type=counterfactual prompt.type=zs dataset.name=comve \
    model.name=qwen decoding.type=gd

# Counter: filter flipped items
python src/evaluation/counter.py

# Step 2: Generate explanations (on flipped items)
python src/runners/explanation_runner.py \
    --config configs/explanation.yaml \
    dataset.type=counterfactual prompt.type=zs dataset.name=comve \
    model.name=qwen decoding.type=gd

# Steps 3-4: Feedback + Refinement (repeat for iter 0, 1, 2)
python src/runners/feedback_runner.py \
    --config configs/feedback.yaml \
    dataset.type=counterfactual prompt.type=zs dataset.name=comve \
    model.name=qwen feedback.type=nl iteration=0

python src/runners/refinement_runner.py \
    --config configs/refinement.yaml \
    dataset.type=counterfactual prompt.type=zs dataset.name=comve \
    model.name=qwen feedback.type=nl iteration=0

# Evaluate
python src/evaluation/faithfulness.py
```

### PCR Phase 1

```bash
python PCR/src/runners/pcr_phase1_runner.py \
    --model_name qwen \
    --max_iters 3

# Evaluate
python PCR/src/evaluation/faithfulness.py --model_name qwen --phase 1
```

### PCR Phase 2

```bash
python PCR/src/runners/pcr_phase2_runner.py \
    --model_name qwen \
    --max_iters 3 \
    --tau 0.95

# Evaluate both phases together
python PCR/src/evaluation/faithfulness.py --model_name qwen --phase both
```

### View Final Results

```bash
python PCR/pcr_final_results.py
```

---

## Summary: The Improvement Chain

```
SR-NLE Init  →  SR-NLE Best  →  PCR Phase 1  →  PCR Phase 2
  28.09%          36.60%          53.62%           65.53%    [Qwen]
  37.08%          42.13%          38.76%           52.25%    [Falcon]

Unfaithfulness (Qwen):  71.91%  →  63.40%  →  46.38%  →  34.47%
```

**One-line summary (Qwen):**  
*"We reduced unfaithfulness from 71.91% (SR-NLE init) to 34.47% (PCR Phase 2) — a 37.44 percentage point absolute improvement — by replacing shallow word-injection feedback with objective probabilistic margin guidance over contrastive explanation pairs."*

---

*Implemented by: Dibyanayan / Jashneet*  
*Based on: SR-NLE (EMNLP 2025, arXiv:2505.22823)*  
*PCR is an original extension built on top of SR-NLE*

<!-- ## MEMORY OF THIS PROJECT REPO -->
<!-- /home/dibyanayan/.claude/projects/-home-dibyanayan-jashneet/memory/sr-nle.md -->