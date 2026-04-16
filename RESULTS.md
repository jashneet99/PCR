# SR-NLE + PCR Experiment Results
**Dataset: ComVE | Metric: Faithfulness Rate (edit_word present in explanation)**

---

## Counter Rate (Counterfactual Sensitivity)

| Model | Flipped Items | Total | Counter Rate |
|-------|--------------|-------|--------------|
| Qwen  | 235          | 9965  | 2.36%        |
| Falcon| 178          | 9965  | 1.79%        |

> Counter rate = % of counterfactual items where model answer flipped after word edit.
> These flipped items are used as input for explanation + SR-NLE + PCR pipeline.

---

## SR-NLE Results

### Qwen — ComVE (235 items)

| Stage              | Faithful | Total | Faith Rate | Improvement vs Init |
|--------------------|----------|-------|------------|---------------------|
| Init (explanation) | 66       | 235   | 28.09%     | —                   |
| Iter 0 refinement  | 78       | 235   | 33.19%     | +5.10%              |
| Iter 1 refinement  | 85       | 235   | 36.17%     | +8.08%              |
| Iter 2 refinement  | 86       | 235   | 36.60%     | +8.51%              |

**SR-NLE best (Qwen): 28.09% → 36.60% (+8.51% absolute)**

---

### Falcon — ComVE (178 items)

| Stage              | Faithful | Total | Faith Rate | Improvement vs Init |
|--------------------|----------|-------|------------|---------------------|
| Init (explanation) | 66       | 178   | 37.08%     | —                   |
| Iter 0 refinement  | 68       | 178   | 38.20%     | +1.12%              |
| Iter 1 refinement  | 66       | 178   | 37.08%     | +0.00%              |
| Iter 2 refinement  | 75       | 178   | 42.13%     | +5.05%              |

**SR-NLE best (Falcon): 37.08% → 42.13% (+5.05% absolute)**

---

## PCR Results (Qwen — ComVE)

> PCR Phase 1 and Phase 2 both use `explanation_gd.json` (SR-NLE baseline) as input, NOT iter2 output.
> This ensures clean ablation comparison.

| Stage           | Faithful | Total | Faith Rate | Improvement vs SR-NLE baseline | Converged  | Avg Iters |
|-----------------|----------|-------|------------|----------------------------------|------------|-----------|
| SR-NLE baseline | 66       | 235   | 28.09%     | —                                | —          | —         |
| PCR Phase 1     | 126      | 235   | 53.62%     | +25.53%                          | 135/235    | 1.47      |
| PCR Phase 2     | 154      | 235   | 65.53%     | +37.44%                          | 0/235      | 4.00      |

**PCR Phase 1 (Qwen): 28.09% → 53.62% (+25.53% absolute)**
**PCR Phase 2 (Qwen): 28.09% → 65.53% (+37.44% absolute)**

> Note: PCR Phase 2 converged = 0/235 because τ=0.95 threshold is very strict.
> Despite 0 formal convergences, DE refinement via GAP-informed critique still improved faithfulness significantly.
> CE consistently broken (Δ't = -1.0) — model keeps predicting y instead of y' for CE. This is a finding.

---

## PCR Results (Falcon — ComVE)

> TODO: Run PCR Phase 1 and Phase 2 for Falcon

| Stage           | Faith Rate | Status  |
|-----------------|------------|---------|
| SR-NLE baseline | 42.13%     | DONE    |
| PCR Phase 1     | —          | PENDING |
| PCR Phase 2     | —          | PENDING |

---

## Full Comparison Table (Qwen vs Falcon — ComVE)

| Stage               | Qwen Faith Rate | Falcon Faith Rate |
|---------------------|-----------------|-------------------|
| Init (explanation)  | 28.09%          | 37.08%            |
| SR-NLE Iter 2       | 36.60%          | 42.13%            |
| PCR Phase 1         | 53.62%          | PENDING           |
| PCR Phase 2         | 65.53%          | PENDING           |

---

## Key Observations

1. **Falcon starts higher** (37.08% vs 28.09%) — Falcon's initial explanations are more faithful than Qwen's.
2. **SR-NLE helps both** but more for Qwen (+8.51%) than Falcon (+5.05%).
3. **PCR dramatically improves Qwen** — Phase 1 adds +25.53%, Phase 2 adds +37.44% over SR-NLE baseline.
4. **Counter rate**: Qwen (2.36%) > Falcon (1.79%) — Qwen is more sensitive to word edits.
5. **CE breakdown**: In PCR Phase 2, CE margin Δ't = -1.0 consistently — model cannot generate faithful counterfactual explanations. Research finding.
6. **τ=0.95 is strict**: Phase 2 formal convergence = 0/235, but explanation quality still improves through GAP-informed critique.

---

## Pipeline Steps (for reference)

```
Step 1a : answer_runner.py  --  original sentences (1000 items)
Step 1b : answer_runner.py  --  counterfactual sentences (9965 items)
Counter : filter flipped items (answer changed A→B or B→A)
Step 2  : explanation_runner.py  --  generate explanations for flipped items
SR-NLE  : feedback_runner.py + refinement_runner.py  x3 iterations
Eval    : faithfulness.py  --  edit_word in explanation?
PCR P1  : pcr_phase1_runner.py  --  prompt-only CE + Mutual Exclusivity check
PCR P2  : pcr_phase2_runner.py  --  probabilistic margin + GAP-informed critique
```

---

## Models Used

| Model  | HuggingFace ID | Quantization |
|--------|---------------|--------------|
| Qwen   | Qwen/Qwen2.5-7B-Instruct | 8-bit (BitsAndBytes) |
| Falcon | tiiuae/Falcon3-7B-Instruct | 8-bit (BitsAndBytes) |

---

*Last updated: 2026-04-08*
*Future experiments: try on αNLI, e-SNLI datasets; try with Llama, Mistral models*
