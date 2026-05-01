import json
import os
import csv
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_lib
from bert_score import score as bert_score_fn

DATASETS = ['comve', 'ecqa', 'esnli']
MODELS   = ['qwen', 'mistral', 'falcon', 'llama']
STAGES   = ['Init-NLE', 'SR-NLE', 'PCR-1', 'PCR-2']
OUT_DIR  = 'results/Evaluation'

rouge_scorer = rouge_scorer_lib.RougeScorer(['rougeL'], use_stemmer=True)
smoother = SmoothingFunction().method1


def load_gold(dataset):
    data = json.load(open(f'data/counterfactual/{dataset}/gen_final.json'))
    return {item['idx']: item['gold_explanation'] for item in data}


def load_stage(dataset, model, stage, gold):
    """Returns list of (hypothesis, gold_explanation) tuples — all items, no dedup."""
    base     = f'experiments/counterfactual/zs-{dataset}-{model}'
    pcr_base = f'PCR/experiments/counterfactual/zs-{dataset}-{model}'

    def get_gold(r):
        return r.get('gold_explanation') or gold.get(r['idx'])

    if stage == 'Init-NLE':
        data = json.load(open(f'{base}/explanation_gd.json'))
        return [(r['explanation']['final'], get_gold(r))
                for r in data if r.get('explanation', {}).get('final') and get_gold(r)]

    elif stage == 'SR-NLE':
        data = json.load(open(f'{base}/iter2_refinement_nl.json'))
        return [(r['nl_refinement']['final'], get_gold(r))
                for r in data if r.get('nl_refinement', {}).get('final') and get_gold(r)]

    elif stage == 'PCR-1':
        data = json.load(open(f'{pcr_base}/pcr_phase1.json'))
        return [(r['pcr_phase1']['de_final'], get_gold(r))
                for r in data if r.get('pcr_phase1') and r['pcr_phase1'].get('de_final') and get_gold(r)]

    elif stage == 'PCR-2':
        data = json.load(open(f'{pcr_base}/pcr_phase2.json'))
        return [(r['pcr_phase2']['de_final'], get_gold(r))
                for r in data if r.get('pcr_phase2') and r['pcr_phase2'].get('de_final') and get_gold(r)]


def bleu_scores(hyp, ref):
    h = hyp.lower().split()
    r = ref.lower().split()
    b1 = sentence_bleu([r], h, weights=(1,0,0,0),             smoothing_function=smoother)
    b2 = sentence_bleu([r], h, weights=(0.5,0.5,0,0),         smoothing_function=smoother)
    b3 = sentence_bleu([r], h, weights=(1/3,1/3,1/3,0),       smoothing_function=smoother)
    b4 = sentence_bleu([r], h, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother)
    return b1, b2, b3, b4


def rouge_l(hyp, ref):
    return rouge_scorer.score(ref, hyp)['rougeL'].fmeasure


def evaluate_model(dataset, model, gold):
    rows = []

    all_hyps_by_stage = {}
    all_refs_by_stage = {}

    for stage in STAGES:
        try:
            pairs = load_stage(dataset, model, stage, gold)
        except FileNotFoundError:
            print(f"  [SKIP] {dataset}-{model}-{stage} — file missing")
            rows.append({'Stage': stage, 'BLEU-1': None, 'BLEU-2': None,
                         'BLEU-3': None, 'BLEU-4': None, 'ROUGE-L': None, 'BERTScore-F1': None})
            continue

        b1s, b2s, b3s, b4s, rls = [], [], [], [], []
        hyps, refs = [], []

        for hyp, ref in pairs:
            b1, b2, b3, b4 = bleu_scores(hyp, ref)
            rl = rouge_l(hyp, ref)
            b1s.append(b1); b2s.append(b2); b3s.append(b3); b4s.append(b4); rls.append(rl)
            hyps.append(hyp); refs.append(ref)

        all_hyps_by_stage[stage] = hyps
        all_refs_by_stage[stage] = refs

        rows.append({
            'Stage': stage,
            'BLEU-1': round(np.mean(b1s)*100, 2),
            'BLEU-2': round(np.mean(b2s)*100, 2),
            'BLEU-3': round(np.mean(b3s)*100, 2),
            'BLEU-4': round(np.mean(b4s)*100, 2),
            'ROUGE-L': round(np.mean(rls)*100, 2),
            'BERTScore-F1': None
        })

        print(f"  [{dataset}-{model}-{stage}] {len(hyps)} items | "
              f"B1={rows[-1]['BLEU-1']} B4={rows[-1]['BLEU-4']} RL={rows[-1]['ROUGE-L']}")

    # BERTScore — batched per stage
    print(f"  Computing BERTScore for {dataset}-{model}...")
    for i, stage in enumerate(STAGES):
        hyps = all_hyps_by_stage.get(stage, [])
        refs = all_refs_by_stage.get(stage, [])
        if not hyps:
            continue
        _, _, F1 = bert_score_fn(hyps, refs, lang='en', batch_size=64,
                                 model_type='distilbert-base-uncased', verbose=False)
        rows[i]['BERTScore-F1'] = round(F1.mean().item()*100, 2)
        print(f"    {stage} BERTScore-F1={rows[i]['BERTScore-F1']}")

    return rows


def save_per_model(dataset, model, rows):
    path = f'{OUT_DIR}/per_model/{dataset}_{model}_eval.csv'
    fields = ['Stage','BLEU-1','BLEU-2','BLEU-3','BLEU-4','ROUGE-L','BERTScore-F1']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")


def save_macro_avg(dataset, all_model_rows):
    path = f'{OUT_DIR}/macro_avg/{dataset}_macro_avg.csv'
    fields = ['Stage','BLEU-1','BLEU-2','BLEU-3','BLEU-4','ROUGE-L','BERTScore-F1']
    macro_rows = []
    for si, stage in enumerate(STAGES):
        vals = {f: [] for f in fields[1:]}
        for model_rows in all_model_rows:
            row = model_rows[si]
            for f in fields[1:]:
                if row[f] is not None:
                    vals[f].append(row[f])
        macro_rows.append({
            'Stage': stage,
            **{f: round(np.mean(v), 2) if v else None for f, v in vals.items()}
        })
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(macro_rows)
    print(f"  Saved macro avg: {path}")
    return macro_rows


def save_main_paper_table(all_macro):
    path = f'{OUT_DIR}/main_paper_table.csv'
    fields = ['Stage',
              'ComVE_BLEU1','ComVE_BLEU2','ComVE_BLEU3','ComVE_BLEU4','ComVE_ROUGEL','ComVE_BERTScore',
              'eSNLI_BLEU1','eSNLI_BLEU2','eSNLI_BLEU3','eSNLI_BLEU4','eSNLI_ROUGEL','eSNLI_BERTScore',
              'ECQA_BLEU1','ECQA_BLEU2','ECQA_BLEU3','ECQA_BLEU4','ECQA_ROUGEL','ECQA_BERTScore']
    rows = []
    for si, stage in enumerate(STAGES):
        row = {'Stage': stage}
        for ds, prefix in zip(['comve','esnli','ecqa'], ['ComVE','eSNLI','ECQA']):
            macro = all_macro[ds][si]
            row[f'{prefix}_BLEU1']     = macro['BLEU-1']
            row[f'{prefix}_BLEU2']     = macro['BLEU-2']
            row[f'{prefix}_BLEU3']     = macro['BLEU-3']
            row[f'{prefix}_BLEU4']     = macro['BLEU-4']
            row[f'{prefix}_ROUGEL']    = macro['ROUGE-L']
            row[f'{prefix}_BERTScore'] = macro['BERTScore-F1']
        rows.append(row)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\nMain paper table saved: {path}")


if __name__ == '__main__':
    os.makedirs(f'{OUT_DIR}/per_model', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/macro_avg', exist_ok=True)

    all_macro = {}
    for dataset in DATASETS:
        print(f"\n{'='*50}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*50}")
        gold = load_gold(dataset)
        all_model_rows = []
        for model in MODELS:
            print(f"\n  Model: {model}")
            rows = evaluate_model(dataset, model, gold)
            save_per_model(dataset, model, rows)
            all_model_rows.append(rows)
        macro = save_macro_avg(dataset, all_model_rows)
        all_macro[dataset] = macro

    save_main_paper_table(all_macro)
    print("\nDone! All evaluation files saved in results/Evaluation/")
