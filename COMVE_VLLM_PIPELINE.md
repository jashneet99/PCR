# ComVE vLLM Pipeline Commands
Branch: `feature/ecqa` | GPU: `CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48`
Models: **qwen → mistral → falcon** (run one at a time)

> ⚠️ This re-run will overwrite the old HuggingFace+bitsandbytes ComVE results.
> Old results already captured in `results/ComVE/` — safe to proceed.

---

## Setup (run once)
```bash
cd /home/dibyanayan/jashneet/SR-NLE
source sr-nle-env/bin/activate
```

---

## Data — Already Ready ✅
```
data/formatted/comve/test.json                  → 1000 items  (original)
data/counterfactual/comve/ext_org.json          → 9965 items  (counterfactual)
```
No data prep needed — re-use existing files.

---

## PHASE 1 — SR-NLE Pipeline

> Repeat all steps below for each model.
> Replace `model.name=qwen` with `model.name=mistral` or `model.name=falcon`.

---

### Step 1a — Original Answers (1000 items)
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/answer_runner.py \
  --config=configs/answer.yaml \
  dataset.type=original dataset.name=comve \
  prompt.type=zs model.name=qwen decoding.type=gd
```
Output: `experiments/original/zs-comve-qwen/answer_gd.json`

---

### Step 1b — Counterfactual Answers (9965 items)
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/answer_runner.py \
  --config=configs/answer.yaml \
  dataset.type=counterfactual dataset.name=comve \
  prompt.type=zs model.name=qwen decoding.type=gd
```
Output: `experiments/counterfactual/zs-comve-qwen/answer_gd.json`

---

### Step 1c — Filter Flipped Items
```bash
python3 -c "
import json, os
model = 'qwen'
with open(f'experiments/original/zs-comve-{model}/answer_gd.json') as f:
    org = json.load(f)
with open(f'experiments/counterfactual/zs-comve-{model}/answer_gd.json') as f:
    ct = json.load(f)
flipped = [item for item in ct
           if item.get('answer',{}).get('final') is not None
           and org[item['idx']].get('answer',{}).get('final') is not None
           and item['answer']['final'] != org[item['idx']]['answer']['final']]
save = f'experiments/counterfactual/zs-comve-{model}/answer_gd_counter.json'
os.makedirs(os.path.dirname(save), exist_ok=True)
json.dump(flipped, open(save,'w'), indent=4, ensure_ascii=False)
print(f'Flipped: {len(flipped)} / {len(ct)}')
"
```
Output: `experiments/counterfactual/zs-comve-qwen/answer_gd_counter.json`

---

### Step 2 — Explanation Generation
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/explanation_runner.py \
  --config=configs/explanation.yaml \
  dataset.type=counterfactual dataset.name=comve \
  prompt.type=zs model.name=qwen decoding.type=gd
```
Output: `experiments/counterfactual/zs-comve-qwen/explanation_gd.json`

---

### Step 3+4 — Feedback + Refinement (3 iterations)

#### Iteration 0
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/feedback_runner.py \
  --config=configs/feedback.yaml \
  dataset.type=counterfactual dataset.name=comve \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=0

CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/refinement_runner.py \
  --config=configs/refinement.yaml \
  dataset.type=counterfactual dataset.name=comve \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=0
```

#### Iteration 1
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/feedback_runner.py \
  --config=configs/feedback.yaml \
  dataset.type=counterfactual dataset.name=comve \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=1

CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/refinement_runner.py \
  --config=configs/refinement.yaml \
  dataset.type=counterfactual dataset.name=comve \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=1
```

#### Iteration 2
```bash
CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/feedback_runner.py \
  --config=configs/feedback.yaml \
  dataset.type=counterfactual dataset.name=comve \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=2

CUDA_VISIBLE_DEVICES=1 VLLM_GPU_MEM_UTIL=0.48 python src/runners/refinement_runner.py \
  --config=configs/refinement.yaml \
  dataset.type=counterfactual dataset.name=comve \
  prompt.type=zs model.name=qwen feedback.type=nl seed=42 iteration=2
```
Output: `experiments/counterfactual/zs-comve-qwen/explanation_gd.json` (updated with all iterations)

---

## PHASE 2 — PCR Pipeline

> Run AFTER SR-NLE is fully done for the model.

### PCR Phase 1
```bash
CUDA_VISIBLE_DEVICES=1 python PCR/src/runners/pcr_phase1_runner.py \
  --model_name qwen --dataset_name comve --max_iters 3
```
Output: `PCR/experiments/counterfactual/zs-comve-qwen/pcr_phase1.json`

### PCR Phase 2
```bash
CUDA_VISIBLE_DEVICES=1 python PCR/src/runners/pcr_phase2_runner.py \
  --model_name qwen --dataset_name comve --max_iters 3 --tau 0.95
```
Output: `PCR/experiments/counterfactual/zs-comve-qwen/pcr_phase2.json`

---

## PHASE 3 — Faithfulness Evaluation

> Run after PCR Phase 2 is done for the model.

```bash
cd /home/dibyanayan/jashneet/SR-NLE
python src/evaluation/faithfulness.py
python PCR/src/evaluation/faithfulness.py --model_name qwen --phase both
```

---

## Model Order & Progress Tracker

### Qwen
| Step | Command | Status |
|------|---------|--------|
| 1a — original answers | `answer_runner.py ... dataset.type=original ... model.name=qwen` | ⏳ |
| 1b — counterfactual answers | `answer_runner.py ... dataset.type=counterfactual ... model.name=qwen` | ⏳ |
| 1c — filter flipped | inline python (change `model='qwen'`) | ⏳ |
| 2 — explanation | `explanation_runner.py ... model.name=qwen` | ⏳ |
| 3+4 — feedback+refinement iter 0 | feedback + refinement runners | ⏳ |
| 3+4 — feedback+refinement iter 1 | feedback + refinement runners | ⏳ |
| 3+4 — feedback+refinement iter 2 | feedback + refinement runners | ⏳ |
| PCR Phase 1 | `pcr_phase1_runner.py --model_name qwen` | ⏳ |
| PCR Phase 2 | `pcr_phase2_runner.py --model_name qwen` | ⏳ |
| Faithfulness eval | `faithfulness.py` | ⏳ |

### Mistral
| Step | Status |
|------|--------|
| All steps (replace `qwen` → `mistral`) | ⏳ |

### Falcon
| Step | Status |
|------|--------|
| All steps (replace `qwen` → `falcon`) | ⏳ |

---

## Quick Reference — Change Model Name
To run for a different model, replace in every command:
- `model.name=qwen`  → `model.name=mistral` or `model.name=falcon`
- `--model_name qwen` → `--model_name mistral` or `--model_name falcon`
- `model = 'qwen'` (in inline python) → `'mistral'` or `'falcon'`
