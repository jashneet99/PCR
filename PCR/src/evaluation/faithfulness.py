"""
PCR Faithfulness Evaluation — Phase 1 and Phase 2
----------------------------------------------------
Evaluates pcr_phase1.json and pcr_phase2.json using the same metric as SR-NLE:
    Faith Rate = % of explanations that mention the edit_word

For Phase 2, also reports average final Δt and Δ't margins.
"""

import json
import os
import argparse
import pandas as pd


def is_word_in_expl(word: str, expl: str) -> bool:
    return word.lower() in expl.lower()


def evaluate_phase1(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = sum(
        1 for item in data
        if item['pcr_phase1']['y'] != item['pcr_phase1']['y_prime']
    )

    base_faith = sum(
        1 for item in data
        if is_word_in_expl(item['edit_word'], item['pcr_phase1']['de_original'])
    )
    pcr_faith = sum(
        1 for item in data
        if is_word_in_expl(item['edit_word'], item['pcr_phase1']['de_final'])
    )
    converged  = sum(1 for item in data if item['pcr_phase1']['converged'])
    avg_iters  = sum(item['pcr_phase1']['iterations'] for item in data) / total

    return {
        "phase": "Phase 1",
        "total": total,
        "base_faith": base_faith,
        "base_faith_rate": round(base_faith / total, 4),
        "pcr_faith": pcr_faith,
        "pcr_faith_rate": round(pcr_faith / total, 4),
        "pcr_unfaith_rate": round((total - pcr_faith) / total, 4),
        "converged": converged,
        "convergence_rate": round(converged / total, 4),
        "avg_iters": round(avg_iters, 2),
    }


def evaluate_phase2(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = sum(
        1 for item in data
        if item['pcr_phase2']['y'] != item['pcr_phase2']['y_prime']
    )

    base_faith = sum(
        1 for item in data
        if is_word_in_expl(item['edit_word'], item['pcr_phase2']['de_original'])
    )
    pcr_faith = sum(
        1 for item in data
        if is_word_in_expl(item['edit_word'], item['pcr_phase2']['de_final'])
    )
    converged     = sum(1 for item in data if item['pcr_phase2']['converged'])
    avg_iters     = sum(item['pcr_phase2']['iterations'] for item in data) / total
    avg_delta_t   = sum(item['pcr_phase2']['final_delta_t'] for item in data) / total
    avg_delta_pt  = sum(item['pcr_phase2']['final_delta_prime_t'] for item in data) / total

    return {
        "phase": "Phase 2",
        "total": total,
        "base_faith": base_faith,
        "base_faith_rate": round(base_faith / total, 4),
        "pcr_faith": pcr_faith,
        "pcr_faith_rate": round(pcr_faith / total, 4),
        "pcr_unfaith_rate": round((total - pcr_faith) / total, 4),
        "converged": converged,
        "convergence_rate": round(converged / total, 4),
        "avg_iters": round(avg_iters, 2),
        "avg_final_delta_t": round(avg_delta_t, 4),
        "avg_final_delta_prime_t": round(avg_delta_pt, 4),
    }


def print_results(r1=None, r2=None):
    print("\n" + "=" * 65)
    print("        PCR Faithfulness Evaluation")
    print("=" * 65)

    print(f"""
  Baseline (SR-NLE init, de_original):
    Faithful     : {r1['base_faith'] if r1 else r2['base_faith']}/235
    Faith Rate   : {r1['base_faith_rate'] if r1 else r2['base_faith_rate']:.4f}
    Unfaith Rate : {1 - (r1['base_faith_rate'] if r1 else r2['base_faith_rate']):.4f}
""")

    print("  " + "-" * 61)
    print(f"""
  Comparison Table:
  +------------------------+------------+------------+------------+
  |  Stage                 | Faith Rate |Unfaith Rate| Converged  |
  +------------------------+------------+------------+------------+
  | SR-NLE baseline        |   {r1['base_faith_rate'] if r1 else r2['base_faith_rate']:.4f}   |   {1-(r1['base_faith_rate'] if r1 else r2['base_faith_rate']):.4f}   |     -      |""")

    if r1:
        print(f"  | PCR Phase 1            |   {r1['pcr_faith_rate']:.4f}   |   {r1['pcr_unfaith_rate']:.4f}   | {r1['converged']:>4}/{r1['total']:<4}  |")
    if r2:
        print(f"  | PCR Phase 2            |   {r2['pcr_faith_rate']:.4f}   |   {r2['pcr_unfaith_rate']:.4f}   | {r2['converged']:>4}/{r2['total']:<4}  |")
    print("  +------------------------+------------+------------+------------+")

    if r1:
        imp1 = (r1['pcr_faith_rate'] - r1['base_faith_rate']) * 100
        print(f"\n  PCR Phase 1 improvement over SR-NLE baseline : {imp1:+.2f}%")
        print(f"  PCR Phase 1 avg iterations : {r1['avg_iters']}")

    if r2:
        imp2 = (r2['pcr_faith_rate'] - r2['base_faith_rate']) * 100
        print(f"\n  PCR Phase 2 improvement over SR-NLE baseline : {imp2:+.2f}%")
        print(f"  PCR Phase 2 avg iterations : {r2['avg_iters']}")
        print(f"  PCR Phase 2 avg final Δt   : {r2['avg_final_delta_t']}")
        print(f"  PCR Phase 2 avg final Δ't  : {r2['avg_final_delta_prime_t']}")

    if r1 and r2:
        imp_diff = (r2['pcr_faith_rate'] - r1['pcr_faith_rate']) * 100
        print(f"\n  Phase 2 vs Phase 1         : {imp_diff:+.2f}%")

    print("=" * 65)


def save_csv(r1, r2, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = [
        {
            "Model": model_name, "Dataset": "comve",
            "Stage": "SR-NLE baseline",
            "Faith Rate": r1['base_faith_rate'] if r1 else r2['base_faith_rate'],
            "Unfaith Rate": 1 - (r1['base_faith_rate'] if r1 else r2['base_faith_rate']),
            "Converged": "-", "Avg Iters": "-",
        }
    ]
    if r1:
        rows.append({
            "Model": model_name, "Dataset": "comve",
            "Stage": "PCR Phase 1",
            "Faith Rate": r1['pcr_faith_rate'],
            "Unfaith Rate": r1['pcr_unfaith_rate'],
            "Converged": f"{r1['converged']}/{r1['total']}",
            "Avg Iters": r1['avg_iters'],
        })
    if r2:
        rows.append({
            "Model": model_name, "Dataset": "comve",
            "Stage": "PCR Phase 2",
            "Faith Rate": r2['pcr_faith_rate'],
            "Unfaith Rate": r2['pcr_unfaith_rate'],
            "Converged": f"{r2['converged']}/{r2['total']}",
            "Avg Iters": r2['avg_iters'],
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "pcr_faithfulness_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen')
    parser.add_argument('--phase', type=str, default='both',
                        choices=['1', '2', 'both'])
    args = parser.parse_args()

    base = f"PCR/experiments/counterfactual/zs-comve-{args.model_name}"
    p1_path = f"{base}/pcr_phase1.json"
    p2_path = f"{base}/pcr_phase2.json"

    r1 = evaluate_phase1(p1_path) if args.phase in ('1', 'both') and os.path.exists(p1_path) else None
    r2 = evaluate_phase2(p2_path) if args.phase in ('2', 'both') and os.path.exists(p2_path) else None

    if r1 is None and r2 is None:
        print("No output files found. Run the Phase 1 or Phase 2 runner first.")
        exit(1)

    print_results(r1, r2)
    save_csv(r1, r2, args.model_name, "PCR/logs/df")
