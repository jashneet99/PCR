# Faithfulness Results — SR-NLE & PCR

> **Metric:** Faith Rate = % of explanations that mention the edit/key word  
> **SR-NLE** = best iteration (Iter 2) | **PCR-1** = Phase 1 | **PCR-2** = Phase 2  
> **—** = not evaluated for this dataset/model combination

---

## ComVE

| Stage     |  Qwen  | Mistral | Falcon | Llama |
|-----------|:------:|:-------:|:------:|:-----:|
| Init-NLE  | 28.09% | 28.57%  | 37.08% |   —   |
| SR-NLE    | 36.60% | 37.18%  | 42.13% |   —   |
| PCR-1     | 53.62% | 41.81%  | 38.76% |   —   |
| **PCR-2** | **65.53%** | **54.41%** | **52.25%** | **—** |

---

## eSNLI

| Stage     |  Qwen  | Mistral | Falcon | Llama  |
|-----------|:------:|:-------:|:------:|:------:|
| Init-NLE  | 59.18% | 54.31%  | 82.11% | 48.47% |
| SR-NLE    | 64.49% | 55.03%  | 77.72% | 56.37% |
| PCR-1     | 70.32% | 70.92%  | 81.63% | 63.57% |
| **PCR-2** | **89.74%** | **73.92%** | **84.78%** | **66.95%** |

---

## ECQA

| Stage     |  Qwen  | Mistral | Falcon | Llama  |
|-----------|:------:|:-------:|:------:|:------:|
| Init-NLE  | 55.84% | 59.10%  | 54.38% | 57.44% |
| SR-NLE    | 60.65% | 57.87%  | 57.40% | 62.36% |
| PCR-1     | 70.35% | 64.26%  | 57.46% | 61.01% |
| **PCR-2** | **74.33%** | **67.98%** | **66.71%** | **63.30%** |

---

## PCR-2 Gain over Init-NLE (absolute %)

| Dataset | Qwen   | Mistral | Falcon | Llama  |
|---------|:------:|:-------:|:------:|:------:|
| ComVE   | +37.44 | +25.84  | +15.17 |   —    |
| eSNLI   | +30.56 | +19.61  | +2.67  | +18.48 |
| ECQA    | +18.49 | +8.88   | +12.33 | +5.86  |
