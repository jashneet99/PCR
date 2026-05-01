import os
import json
import csv
import re
import numpy as np
from vllm import LLM, SamplingParams

DATASETS = ['comve', 'ecqa', 'esnli']
MODELS   = ['qwen', 'mistral', 'falcon', 'llama']
STAGES   = ['Init-NLE', 'SR-NLE', 'PCR-1', 'PCR-2']
OUT_DIR  = 'results/LLM_Judge'
JUDGE_MODEL = 'prometheus-eval/prometheus-7b-v2.0'

# ── Prometheus-2 prompt template ──────────────────────────────────────────────
PROMETHEUS_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, without evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback:"""

PLAUSIBILITY_RUBRIC = (
    "[Is the explanation plausible — does it logically justify the model's prediction "
    "given the counterfactual input?]\n"
    "Score 1: The explanation contradicts the input or is entirely irrelevant to the prediction.\n"
    "Score 2: The explanation is loosely related to the input but does not justify the prediction.\n"
    "Score 3: The explanation partially justifies the prediction but misses key reasoning.\n"
    "Score 4: The explanation mostly justifies the prediction with only minor gaps.\n"
    "Score 5: The explanation clearly and logically justifies the model's prediction based on the input."
)

NATURALNESS_RUBRIC = (
    "[Is the explanation natural — does it read as fluent, coherent, and human-like text?]\n"
    "Score 1: The explanation is incoherent, grammatically broken, or unreadable.\n"
    "Score 2: The explanation is understandable but unnatural, repetitive, or robotic.\n"
    "Score 3: The explanation is somewhat natural but has noticeable awkward phrasing.\n"
    "Score 4: The explanation reads naturally with only minor stylistic issues.\n"
    "Score 5: The explanation reads perfectly natural, fluent, and coherent like human writing."
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def letter_to_choice(letter, choices):
    """Map answer letter (A/B/C/D/E) to actual choice text."""
    idx = ord(letter.upper()) - ord('A')
    if 0 <= idx < len(choices):
        return choices[idx]
    return letter


def build_instruction(dataset, item):
    answer_letter = (item.get('answer') or {}).get('final', '') \
        if isinstance(item.get('answer'), dict) else ''
    choices = item.get('choices', [])
    answer_text = letter_to_choice(answer_letter, choices) if answer_letter else ''

    if dataset == 'comve':
        s0       = item.get('sentence0', '')
        s1       = item.get('sentence1', '')
        edit_pos = item.get('edit_pos', '')
        edit_word = item.get('edit_word', '')
        return (
            f"Two sentences are given. One violates commonsense due to a word substitution.\n"
            f"Sentence 0: {s0}\n"
            f"Sentence 1: {s1}\n"
            f"Edit word inserted: '{edit_word}' (in {edit_pos})\n"
            f"The model predicted '{answer_text}' as the commonsense-violating sentence.\n"
            f"Provide a natural language explanation for why this sentence violates commonsense."
        )

    elif dataset == 'ecqa':
        question    = item.get('question', '')
        choices_str = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
        return (
            f"Question: {question}\n"
            f"Answer Choices:\n{choices_str}\n"
            f"The model selected: ({answer_letter}) {answer_text}\n"
            f"Provide an explanation for why this answer is correct."
        )

    elif dataset == 'esnli':
        premise    = item.get('premise', '')
        hypothesis = item.get('hypothesis', '')
        return (
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            f"The model predicted the relationship as: {answer_text}\n"
            f"Provide an explanation for this relationship."
        )


def make_prompt(dataset, item, explanation, gold_explanation, rubric):
    instruction = build_instruction(dataset, item)
    return PROMETHEUS_TEMPLATE.format(
        instruction=instruction,
        response=explanation,
        reference_answer=gold_explanation,
        rubric=rubric
    )


def parse_score(text):
    m = re.search(r'\[RESULT\]\s*([1-5])', text)
    if m:
        return int(m.group(1))
    # Fallback: last digit 1-5 in output
    m = re.search(r'([1-5])\s*$', text.strip())
    if m:
        return int(m.group(1))
    return None


# ── Data loading ──────────────────────────────────────────────────────────────
def load_gold(dataset):
    data = json.load(open(f'data/counterfactual/{dataset}/gen_final.json'))
    return {item['idx']: item for item in data}


def load_stage(dataset, model, stage, gold):
    """Returns list of (explanation, gold_explanation, enriched_item) for flipped items."""
    base     = f'experiments/counterfactual/zs-{dataset}-{model}'
    pcr_base = f'PCR/experiments/counterfactual/zs-{dataset}-{model}'

    def get_gold_exp(r):
        return r.get('gold_explanation') or (gold.get(r['idx']) or {}).get('gold_explanation')

    def enrich(r):
        # Merge gold fields (context) with generated fields; generated fields take priority
        g = dict(gold.get(r['idx'], {}))
        g.update(r)
        return g

    if stage == 'Init-NLE':
        data = json.load(open(f'{base}/explanation_gd.json'))
        return [(r['explanation']['final'], get_gold_exp(r), enrich(r))
                for r in data if r.get('explanation', {}).get('final') and get_gold_exp(r)]

    elif stage == 'SR-NLE':
        data = json.load(open(f'{base}/iter2_refinement_nl.json'))
        return [(r['nl_refinement']['final'], get_gold_exp(r), enrich(r))
                for r in data if r.get('nl_refinement', {}).get('final') and get_gold_exp(r)]

    elif stage == 'PCR-1':
        data = json.load(open(f'{pcr_base}/pcr_phase1.json'))
        return [(r['pcr_phase1']['de_final'], get_gold_exp(r), enrich(r))
                for r in data if r.get('pcr_phase1') and r['pcr_phase1'].get('de_final') and get_gold_exp(r)]

    elif stage == 'PCR-2':
        data = json.load(open(f'{pcr_base}/pcr_phase2.json'))
        return [(r['pcr_phase2']['de_final'], get_gold_exp(r), enrich(r))
                for r in data if r.get('pcr_phase2') and r['pcr_phase2'].get('de_final') and get_gold_exp(r)]


# ── Per-stage checkpoint helpers ──────────────────────────────────────────────
def stage_cache_path(dataset, model, stage):
    key = stage.replace('-', '_').replace(' ', '_')
    return f'{OUT_DIR}/stage_cache/{dataset}_{model}_{key}.json'


def load_stage_cache(dataset, model, stage):
    path = stage_cache_path(dataset, model, stage)
    if os.path.exists(path):
        return json.load(open(path))
    return None


def save_stage_cache(dataset, model, stage, row):
    path = stage_cache_path(dataset, model, stage)
    with open(path, 'w') as f:
        json.dump(row, f)


# ── Judge one model ───────────────────────────────────────────────────────────
def judge_model(llm, sampling_params, dataset, model, gold):
    rows = []

    for stage in STAGES:
        # Resume from per-stage checkpoint if available
        cached = load_stage_cache(dataset, model, stage)
        if cached is not None:
            print(f"  [SKIP stage] {dataset}-{model}-{stage} — loaded from cache")
            rows.append(cached)
            continue

        try:
            triples = load_stage(dataset, model, stage, gold)
        except FileNotFoundError:
            print(f"  [SKIP] {dataset}-{model}-{stage} — file missing")
            row = {'Stage': stage, 'Plausibility': None, 'Naturalness': None}
            rows.append(row)
            save_stage_cache(dataset, model, stage, row)
            continue

        p_prompts = [make_prompt(dataset, item, exp, gexp, PLAUSIBILITY_RUBRIC)
                     for exp, gexp, item in triples]
        n_prompts = [make_prompt(dataset, item, exp, gexp, NATURALNESS_RUBRIC)
                     for exp, gexp, item in triples]

        print(f"  [{dataset}-{model}-{stage}] {len(triples)} items — running judge...")

        p_outputs = llm.generate(p_prompts, sampling_params)
        n_outputs = llm.generate(n_prompts, sampling_params)

        p_scores = [parse_score(o.outputs[0].text) for o in p_outputs]
        n_scores = [parse_score(o.outputs[0].text) for o in n_outputs]

        p_valid = [s for s in p_scores if s is not None]
        n_valid = [s for s in n_scores if s is not None]

        p_avg = round(np.mean(p_valid), 3) if p_valid else None
        n_avg = round(np.mean(n_valid), 3) if n_valid else None

        row = {'Stage': stage, 'Plausibility': p_avg, 'Naturalness': n_avg}
        rows.append(row)
        save_stage_cache(dataset, model, stage, row)   # checkpoint after each stage
        print(f"    Plausibility={p_avg} (parsed {len(p_valid)}/{len(p_scores)}) | "
              f"Naturalness={n_avg} (parsed {len(n_valid)}/{len(n_scores)})")

    return rows


# ── Save helpers ──────────────────────────────────────────────────────────────
def save_per_model(dataset, model, rows):
    path = f'{OUT_DIR}/per_model/{dataset}_{model}_judge.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['Stage', 'Plausibility', 'Naturalness'])
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def load_per_model_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return [
            {k: (v if k == 'Stage' else (float(v) if v not in ('', 'None') else None))
             for k, v in r.items()}
            for r in reader
        ]


def save_macro_avg(dataset, all_model_rows):
    path = f'{OUT_DIR}/macro_avg/{dataset}_macro_avg.csv'
    macro_rows = []
    for si, stage in enumerate(STAGES):
        p_vals, n_vals = [], []
        for model_rows in all_model_rows:
            row = model_rows[si]
            if row['Plausibility'] is not None: p_vals.append(row['Plausibility'])
            if row['Naturalness']  is not None: n_vals.append(row['Naturalness'])
        macro_rows.append({
            'Stage':        stage,
            'Plausibility': round(np.mean(p_vals), 3) if p_vals else None,
            'Naturalness':  round(np.mean(n_vals), 3) if n_vals else None,
        })
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['Stage', 'Plausibility', 'Naturalness'])
        w.writeheader()
        w.writerows(macro_rows)
    print(f"  Saved macro avg: {path}")
    return macro_rows


def save_main_table(all_macro):
    path = f'{OUT_DIR}/main_paper_table.csv'
    fields = ['Stage',
              'ComVE_Plausibility', 'ComVE_Naturalness',
              'eSNLI_Plausibility', 'eSNLI_Naturalness',
              'ECQA_Plausibility',  'ECQA_Naturalness']
    rows = []
    for si, stage in enumerate(STAGES):
        row = {'Stage': stage}
        for ds, prefix in zip(['comve', 'esnli', 'ecqa'], ['ComVE', 'eSNLI', 'ECQA']):
            m = all_macro[ds][si]
            row[f'{prefix}_Plausibility'] = m['Plausibility']
            row[f'{prefix}_Naturalness']  = m['Naturalness']
        rows.append(row)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nMain paper table saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(f'{OUT_DIR}/per_model', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/macro_avg', exist_ok=True)

    print(f"Loading judge model: {JUDGE_MODEL}")
    llm = LLM(
        model=JUDGE_MODEL,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    all_macro = {}
    for dataset in DATASETS:
        print(f"\n{'='*55}\nDATASET: {dataset.upper()}\n{'='*55}")
        gold = load_gold(dataset)
        all_model_rows = []

        for model in MODELS:
            print(f"\n  Model: {model}")
            path = f'{OUT_DIR}/per_model/{dataset}_{model}_judge.csv'
            if os.path.exists(path):
                print(f"  [SKIP] Already exists: {path}")
                all_model_rows.append(load_per_model_csv(path))
                continue
            rows = judge_model(llm, sampling_params, dataset, model, gold)
            save_per_model(dataset, model, rows)
            all_model_rows.append(rows)

        macro = save_macro_avg(dataset, all_model_rows)
        all_macro[dataset] = macro

    save_main_table(all_macro)
    print("\nDone! All LLM Judge files saved in results/LLM_Judge/")
