# Faithfulness Tables
> Metric: Faith Rate = % of explanations that mention the edit/key word
> SR-NLE = Iter 2 (best iteration)

---

## Experimental Details

All experiments were conducted in a zero-shot, inference-only setting using frozen instruction-tuned LLMs. Since no model fine-tuning was performed, training-specific hyperparameters such as optimizer and learning rate are not applicable. Table 1 summarizes the key implementation, inference, hardware, and evaluation hyperparameters used in our experiments.

| **Category** | **Setting Used in Experiments** |
|---|---|
| Training / fine-tuning | None; all models are used as frozen zero-shot generators. |
| Optimizer | Not applicable; no model parameters are updated. |
| Learning rate | Not applicable. |
| Models | Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Falcon3-7B-Instruct, Llama-3.1-8B-Instruct. |
| Inference backend | vLLM. |
| GPU configuration | Single NVIDIA A100 80GB PCIe GPU per run; `tensor_parallel_size=1`. |
| vLLM GPU memory utilization | `VLLM_GPU_MEM_UTIL=0.48` for the reported pipeline runs. |
| Model dtype | `bfloat16`. |
| Generation batch size | Dynamic vLLM batching; no fixed manual generation batch size is set. |
| Original sample size | 1,000 test examples per dataset. |
| Counterfactual candidates | Up to 10 edits per original example, then filtered to answer-flipping counterfactuals. |
| Decoding used for reported main results | Greedy decoding (`do_sample=false`). |
| Greedy decoding parameters | `temperature=0.0`, `top_p=1.0`, `top_k=-1`, `num_return_sequences=1`. |
| Maximum generation length | 512 new tokens for answer, explanation, feedback, and SR-NLE refinement stages. PCR uses shorter task-specific limits: 10 tokens for margin/evidence checks, 150 tokens for counter-explanations and Phase 1 critiques, and 200 tokens for Phase 2 critique/refinement prompts. |
| Self-consistency setting available in configs | Sampling with `temperature=1.0`, `num_return_sequences=20`, `top_p=1.0`, `top_k=-1`; majority voting for answers and random selection for explanation/feedback/refinement. Not used for the greedy reported tables unless explicitly run with `decoding.type=sc`. |
| SR-NLE refinement iterations | Three refinement rounds are run (`iter0`, `iter1`, `iter2`); reported SR-NLE results use `iter2`. |
| Attribution feedback | Integrated Gradients with EOS-token baseline, target aggregation `abs_mean`, word aggregation `sum`. |
| Integrated Gradients steps | Qwen: 1,000 steps; Mistral, Falcon, and Llama: 500 steps. |
| Integrated Gradients internal batch size | Starts at 50 and halves on CUDA out-of-memory until it succeeds. |
| Attribution top-k words | Top 5 important words are passed to refinement for attribution-based feedback. |
| Automatic evaluation batch size | BERTScore uses batch size 64 with `distilbert-base-uncased`. |
| LLM judge settings | vLLM judge inference with `temperature=0.0`, `max_tokens=512`, `tensor_parallel_size=1`, and GPU memory utilization `0.85`. |

---

## ComVE

| **Stage** |  Qwen  | Mistral | Falcon |  Llama  |
|-----------|:------:|:-------:|:------:|:-------:|
| Init-NLE  | 29.49% | 37.58%  | 40.99% | 33.33%  |
| SR-NLE    | 31.80% | 43.18%  | 45.34% | 40.63%  |
| PCR-1     | 53.46% | 46.67%  | 45.34% | 40.20%  |
| PCR-2     | 62.67% | 59.70%  | 52.17% | 43.06%  |

---

## eSNLI

| **Stage** |  Qwen  | Mistral | Falcon |  Llama  |
|-----------|:------:|:-------:|:------:|:-------:|
| Init-NLE  | 59.18% | 54.31%  | 82.11% | 48.47%  |
| SR-NLE    | 64.49% | 55.03%  | 77.72% | 56.37%  |
| PCR-1     | 70.32% | 70.92%  | 81.63% | 63.57%  |
| PCR-2     | 89.74% | 73.92%  | 84.78% | 66.95%  |

---

## ECQA

| **Stage** |  Qwen  | Mistral | Falcon |  Llama  |
|-----------|:------:|:-------:|:------:|:-------:|
| Init-NLE  | 55.84% | 59.10%  | 54.38% | 57.44%  |
| SR-NLE    | 60.65% | 57.87%  | 57.40% | 62.36%  |
| PCR-1     | 70.35% | 64.26%  | 57.46% | 61.01%  |
| PCR-2     | 74.33% | 67.98%  | 66.71% | 63.30%  |

---

## Qualitative Case Study

Table 2 presents representative examples comparing Init-NLE, SR-NLE, and PCR-2. We select cases where the counterfactual input remains semantically coherent, the edit word is absent from both Init-NLE and SR-NLE, and PCR-2 explicitly incorporates the edit word while preserving a plausible and natural explanation for the prediction. These examples illustrate that PCR-2 is not merely longer; it more directly connects the decision-relevant counterfactual detail to the model's answer.

| **Dataset / Model** | **Counterfactual Input** | **Prediction** | **Edit Word** | **Init-NLE** | **SR-NLE** | **PCR-2** | **Why PCR-2 is Better** |
|---|---|---|---|---|---|---|---|
| ComVE / Qwen | Sentence 0: Food gets rapidly hotter as you leave it out.<br>Sentence 1: Food gets cooler as you leave it out. | A | rapidly | Food typically cools down over time when left out, not gets hotter. | Food typically cools down over time when left out at room temperature, not gets hotter. | The statement in (A) that "Food gets rapidly hotter as you leave it out" is incorrect because it contradicts the fundamental principle of heat transfer. When food is left out, it loses heat to its surroundings, not gains heat. This process is governed by the second law of thermodynamics, which states that heat naturally flows from hotter objects to cooler objects. Therefore, the food cools down over time as it loses heat to the surrounding environment, not as it absorbs more heat. This directly contradicts the claim in (A) and aligns with the correct explanation that food gets cooler as you leave it out, as stated in (B). | Init-NLE and SR-NLE correctly mention that food cools down, but they omit the edit word rapidly. PCR-2 explicitly uses rapidly and explains why rapid heating is incompatible with heat transfer, making the explanation more faithful to the counterfactual input. |
| ECQA / Qwen | Question: Where are magazines rarely old?<br>Choices: library, market, doctor, table, library | B | rarely | Markets typically stock new and current magazines to attract customers, making old magazines rare there. | Markets typically stock new magazines to attract customers, making old magazines rare due to quick turnover. | Markets typically stock new and current magazines to attract customers, making old magazines rare there. Unlike libraries, which often keep old magazines for historical or archival purposes, markets focus on providing the latest issues to keep their offerings fresh and appealing to a broad customer base. This ensures that old magazines are rarely found in markets, while libraries and other options like tables or doctor's offices do not typically display or stock magazines, further supporting the rarity of old magazines in markets. | Init-NLE and SR-NLE use the related adjective rare, but do not include the exact counterfactual cue rarely. PCR-2 incorporates rarely naturally and contrasts markets with libraries, giving a clearer justification for answer B. |
| eSNLI / Qwen | Premise: An old man with a package poses in front of an advertisement.<br>Hypothesis: A tall man poses in front of an ad. | B | tall | The premise does not provide information about the man's height, so the hypothesis cannot be confirmed or denied based on the given premise. | The premise does not provide any information about the man's height, so the hypothesis cannot be confirmed or denied based on the given premise. | The premise states that an old man with a package poses in front of an advertisement, while the hypothesis claims a tall man poses in front of an ad. The premise does not provide any information about the man's height, making it impossible to confirm or deny whether the man is tall. Therefore, the hypothesis introduces new information (the man's height) that is not present in the premise and cannot be inferred from it. Since the premise does not provide any information that would either entail or contradict the hypothesis, the relationship between the premise and the hypothesis is neutral. Thus, (B) is the correct NLI relationship. | PCR-2 explicitly grounds the neutral label in the edited attribute tall. Init-NLE and SR-NLE identify missing height information, but PCR-2 more clearly states that tall is newly introduced information that cannot be inferred from the premise. |

---

## Error Analysis

Table 3 shows representative cases for two remaining limitations. The first example illustrates a setting where SR-NLE is already faithful, so PCR-2 provides only marginal additional benefit. The second example illustrates that a faithful PCR-2 explanation may diverge from short reference rationales and therefore receive lower reference-overlap scores.

| **Dataset / Model** | **Counterfactual Input** | **Prediction** | **Edit Word** | **Init-NLE** | **SR-NLE** | **PCR-2** | **Why PCR-2 is Failing** |
|---|---|---|---|---|---|---|---|
| ECQA / Llama | Question: Why does she sing sweetly so much?<br>Choices: insane, annoy people around, enjoy, inhale, had good voice | E | sweetly | She sings sweetly because she has a good voice, which allows her to produce pleasant and melodious sounds when singing. This is the most logical reason for her singing sweetly, as it is a characteristic of people with good vocal abilities. | She sings sweetly because she has a good voice, which allows her to produce pleasant and melodious sounds when singing. This is the most logical reason for her singing sweetly, as a good voice is typically associated with a pleasing and aesthetically pleasing sound that can evoke emotions and create a positive atmosphere, making her singing enjoyable for others to listen to. The other options, such as being insane, annoying people, inhaling, or having no apparent reason, are implausible or less likely explanations for singing sweetly, making "Had good voice" the most plausible choice by process of elimination. | She sings sweetly so much because she has a good voice, which is evident from the fact that her singing is consistently met with positive reactions from others, such as smiles and appreciation, even in situations where she's singing alone or with people who may not be familiar with her voice. This suggests that her pleasant voice is not just a subjective experience, but an objective quality that is perceived by others as well. Furthermore, her ability to produce a wide range of tones and pitches with ease and control, as well as her consistent ability to hit the right notes, are all hallmarks of a skilled singer, which is a key characteristic of someone with a good voice. In contrast, simply enjoying singing (option C) does not necessarily require a good voice, as people with average or poor voices can also enjoy singing for its own sake. | This case falls under limitation 1. SR-NLE already includes the edit word sweetly and gives a faithful explanation linking it to the answer had good voice. PCR-2 mostly elaborates the same reasoning, so probability-guided refinement adds little beyond length and contrastive detail. This mirrors the small aggregate gain on ECQA-Llama, where PCR-2 only slightly improves over SR-NLE. |
| ComVE / Qwen | Sentence 0: Food gets rapidly hotter as you leave it out.<br>Sentence 1: Food gets cooler as you leave it out. | A | rapidly | Food typically cools down over time when left out, not gets hotter. | Food typically cools down over time when left out at room temperature, not gets hotter. | The statement in (A) that "Food gets rapidly hotter as you leave it out" is incorrect because it contradicts the fundamental principle of heat transfer. When food is left out, it loses heat to its surroundings, not gains heat. This process is governed by the second law of thermodynamics, which states that heat naturally flows from hotter objects to cooler objects. Therefore, the food cools down over time as it loses heat to the surrounding environment, not as it absorbs more heat. This directly contradicts the claim in (A) and aligns with the correct explanation that food gets cooler as you leave it out, as stated in (B). | This case falls under limitation 2. The gold/reference rationale is short: "Being outside makes food cool down." PCR-2 is more faithful to the model's contrastive decision because it mentions rapidly and explains the heat-transfer contradiction, but its longer decision-specific wording has lower lexical overlap with the reference. Thus, PCR-2 may appear worse under BLEU, ROUGE-L, or BERTScore even when it better captures the model's actual evidence. |
