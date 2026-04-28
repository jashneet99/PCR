# ECQA Pipeline Commands
Branch: `feature/ecqa` | GPU: `CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48`

## Setup
```bash
cd /home/dibyanayan/jashneet/SR-NLE
source /home/dibyanayan/jashneet/SR-NLE/sr-nle-env/bin/activate
```

---

## Phase 0 — Data Prep ✅ DONE

```bash
# Already completed — do not re-run
# data/formatted/ecqa/test.json          → 1000 items
# data/counterfactual/ecqa/ext_org.json  → 9980 items (10 edits × 998 questions)
# data/counterfactual/ecqa/ext_final.json → 9980 items
```

---

## Phase 1 — SR-NLE Pipeline (run for each model: qwen → mistral → llama → falcon)

### Step 1a — Original answers (1000 items)
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/answer_runner.py \
  --config=configs/answer.yaml \
  dataset.type=original dataset.name=ecqa \
  prompt.type=zs model.name=qwen decoding.type=gd
```

### Step 1b — Counterfactual answers (9980 items)
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/answer_runner.py \
  --config=configs/answer.yaml \
  dataset.type=counterfactual dataset.name=ecqa \
  prompt.type=zs model.name=qwen decoding.type=gd
```

### Step 1c — Filter flipped items (counter.py)
```bash
python src/evaluation/counter.py && cat $(ls -t logs/counter*.log | head -1)
```
Output: `experiments/counterfactual/zs-ecqa-qwen/answer_gd_counter.json`

### Step 2 — Explanation generation
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/explanation_runner.py \
  --config=configs/explanation.yaml \
  dataset.type=counterfactual dataset.name=ecqa \
  prompt.type=zs model.name=qwen decoding.type=gd
```
Output: `experiments/counterfactual/zs-ecqa-qwen/explanation_gd.json`

### Step 3+4 — Feedback + Refinement (3 iterations × feedback type: nl)

#### Iteration 0
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/feedback_runner.py \
  --config=configs/feedback.yaml \
  dataset.type=counterfactual dataset.name=ecqa \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=0

CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/refinement_runner.py \
  --config=configs/refinement.yaml \
  dataset.type=counterfactual dataset.name=ecqa \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=0
```

#### Iteration 1
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/feedback_runner.py \
  --config=configs/feedback.yaml \
  dataset.type=counterfactual dataset.name=ecqa \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=1

CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/refinement_runner.py \
  --config=configs/refinement.yaml \
  dataset.type=counterfactual dataset.name=ecqa \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=1
```

#### Iteration 2
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/feedback_runner.py \
  --config=configs/feedback.yaml \
  dataset.type=counterfactual dataset.name=ecqa \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=2

CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/refinement_runner.py \
  --config=configs/refinement.yaml \
  dataset.type=counterfactual dataset.name=ecqa \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=2
```

### Step 5 — Faithfulness Evaluation
```bash
python src/evaluation/faithfulness.py && cat $(ls -t logs/faithfulness*.log | head -1)
```

---

## Repeat Phase 1 for: mistral, llama, falcon
Replace `model.name=qwen` with `model.name=mistral`, `model.name=llama`, `model.name=falcon`

---

## Phase 2 — PCR Pipeline (after SR-NLE done for all models)

> NOTE: PCR needs `PCR/src/prompts/ecqa_prompts.py` — must be written first.

### PCR Phase 1
```bash
CUDA_VISIBLE_DEVICES=1 python PCR/src/runners/pcr_phase1_runner.py \
  --model_name qwen --max_iters 3
```
Output: `PCR/experiments/counterfactual/zs-ecqa-qwen/pcr_phase1.json`

### PCR Phase 2
```bash
CUDA_VISIBLE_DEVICES=1 python PCR/src/runners/pcr_phase2_runner.py \
  --model_name qwen --max_iters 4 --tau 0.95
```
Output: `PCR/experiments/counterfactual/zs-ecqa-qwen/pcr_phase2.json`

---

## Progress Tracker

### Qwen
| Step | Status |
|------|--------|
| Step 1a — original answers | ✅ Done |
| Step 1b — counterfactual answers | ✅ Done |
| Step 1c — counter.py (1558 flipped) | ✅ Done |
| Step 2 — explanation_runner | ✅ Done |
| Step 3+4 — feedback+refinement iter 0 | ⏳ |
| Step 3+4 — feedback+refinement iter 1 | ⏳ |
| Step 3+4 — feedback+refinement iter 2 | ⏳ |
| Step 5 — faithfulness.py | ⏳ |

### Mistral
| Step | Status |
|------|--------|
| All steps | ⏳ |

### Llama
| Step | Status |
|------|--------|
| All steps | ⏳ |

### Falcon
| Step | Status |
|------|--------|
| All steps | ⏳ |
