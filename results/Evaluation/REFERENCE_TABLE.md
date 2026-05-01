# Evaluation Reference Table

## What We Are Evaluating
Generated explanations at each pipeline stage vs. gold human-written explanations.

---

## Stage → Source File Mapping

| Stage | Source File | Field Used | Description |
|-------|------------|------------|-------------|
| Init-NLE | `experiments/counterfactual/zs-{dataset}-{model}/explanation_gd.json` | `explanation.final` | Initial explanation before any refinement |
| SR-NLE | `experiments/counterfactual/zs-{dataset}-{model}/iter2_refinement_nl.json` | `nl_refinement.final` | Explanation after 2 SR-NLE iterations |
| PCR-1 | `PCR/experiments/counterfactual/zs-{dataset}-{model}/pcr_phase1.json` | `pcr_phase1.de_final` | Explanation after PCR Phase 1 (mutual exclusivity) |
| PCR-2 | `PCR/experiments/counterfactual/zs-{dataset}-{model}/pcr_phase2.json` | `pcr_phase2.de_final` | Explanation after PCR Phase 2 (probability margin) |

---

## Reference (Gold) Explanations

| Dataset | Gold File | Field | Total Items |
|---------|----------|-------|-------------|
| ComVE | `data/counterfactual/comve/gen_final.json` | `gold_explanation` | 1000 |
| ECQA | `data/counterfactual/ecqa/gen_final.json` | `gold_explanation` | 1000 |
| eSNLI | `data/counterfactual/esnli/gen_final.json` | `gold_explanation` | 1000 |

> Note: We evaluate only on the **flipped items** (model changed answer after counterfactual edit).
> Items are matched by `idx` between generated and gold files.

---

## Metrics

| Metric | Variant | What It Measures |
|--------|---------|-----------------|
| BLEU-1 | 1-gram precision | Word-level overlap |
| BLEU-2 | 2-gram precision | Phrase-level overlap |
| BLEU-3 | 3-gram precision | Longer phrase overlap |
| BLEU-4 | 4-gram precision | Full phrase overlap |
| ROUGE-L | Longest Common Subsequence | Recall-oriented fluency |
| BERTScore-F1 | Contextual embeddings | Semantic similarity |

---

## Output Folder Structure

```
results/Evaluation/
├── REFERENCE_TABLE.md               ← this file
├── per_model/
│   ├── comve_qwen_eval.csv          ← per model results
│   ├── comve_mistral_eval.csv
│   ├── comve_falcon_eval.csv
│   ├── comve_llama_eval.csv
│   ├── ecqa_qwen_eval.csv
│   ├── ... (12 files total)
└── macro_avg/
    ├── comve_macro_avg.csv          ← averaged across 4 models (main paper table)
    ├── ecqa_macro_avg.csv
    └── esnli_macro_avg.csv
```

---

## Models Evaluated
- Qwen (Qwen2.5-7B)
- Mistral (Mistral-7B)
- Falcon (Falcon-7B)
- Llama (LLaMA-3.1-8B)

## Datasets
- **ComVE** — Commonsense Validation and Explanation
- **ECQA** — Explanation for CommonsenseQA
- **eSNLI** — Explanations for Stanford NLI
