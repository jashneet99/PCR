"""
PCR Phase 2 Runner — Probabilistic PCR
========================================
Uses probability margins (Δt) instead of model's YES/NO judgment.

Starts FRESH from SR-NLE's explanation_gd.json (NOT from Phase 1 output).
This allows clean comparison:
    SR-NLE best  vs  PCR Phase 1  vs  PCR Phase 2

Algorithm (from research document):
    Step 1: Take SR-NLE explanation as DE
            Generate CE (counterfactual explanation)

    Step 2: Feed DE alone to model → compute Δt  = P(y|DE)  - P(y'|DE)
            Feed CE alone to model → compute Δ't = P(y'|CE) - P(y|CE)

    Step 3: Convergence check
            If Δt >= τ AND Δ't >= τ  → converged (faithful)
            If not → compute GAP values → critique with margin numbers → refine

    Step 4: Refinement — rewrite DE and CE using GAP-informed critique
            Repeat up to max_iters

Threshold τ = 0.95

Input  : experiments/counterfactual/zs-comve-{model}/explanation_gd.json
Output : PCR/experiments/counterfactual/zs-comve-{model}/pcr_phase2.json

Run from SR-NLE/ root:
    python PCR/src/runners/pcr_phase2_runner.py \
        --model_name qwen \
        --max_iters 3 \
        --tau 0.95
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from model.model import GenerationModel
from prompts.comve_prompts import (
    get_ce_generation_prompt,
    get_critique_prompt_phase2,
    get_ce_refinement_prompt,
)


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def parse_explanation(text: str) -> str:
    """Extract text after 'Explanation:' label."""
    match = re.search(r"Explanation:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def answer_to_labels(answer_final: str):
    """
    Convert answer final ('A' or 'B') to y and y_prime.
    y       = what model actually predicted
    y_prime = the opposite
    """
    if answer_final.strip().upper() == "A":
        return "A", "B"
    else:
        return "B", "A"


# ------------------------------------------------------------------ #
#  Main PCR Phase 2 Loop                                              #
# ------------------------------------------------------------------ #

def pcr_phase2(item: dict, model: GenerationModel,
               max_iters: int = 3, tau: float = 0.95) -> dict:
    """
    Runs PCR Phase 2 on a single item.

    Phase 2 difference from Phase 1:
        - Instead of asking "are these mutually exclusive?" (YES/NO)
        - Computes actual probability margins:
            Δt  = P(y  | DE) - P(y' | DE)   for DE
            Δ't = P(y' | CE) - P(y  | CE)   for CE
        - Both must reach threshold τ = 0.95
        - If not → tell model exactly how far off (GAP) → refine

    Starts fresh from SR-NLE's explanation (same input as Phase 1).
    """
    sentence0  = item["sentence0"]
    sentence1  = item["sentence1"]
    de         = item["explanation"]["final"]    # SR-NLE's explanation → DE
    y, y_prime = answer_to_labels(item["answer"]["final"])

    # ---------------------------------------------------------------- #
    # Step 1: Generate CE
    # ---------------------------------------------------------------- #
    ce_prompt = get_ce_generation_prompt(sentence0, sentence1, y, y_prime)
    ce_raw    = model.get_generated(ce_prompt, do_sample=False, max_new_tokens=150)[0]
    ce        = parse_explanation(ce_raw)

    history   = []
    converged = False

    # ---------------------------------------------------------------- #
    # Steps 2-4: Iterative probability-guided refinement
    # ---------------------------------------------------------------- #
    for i in range(max_iters):

        # Step 2a: Compute Δt for DE
        # Feed DE alone (no original sentences) → model predicts y or y'
        delta_t = model.get_margin(de, y, y_prime)

        # Step 2b: Compute Δ't for CE
        # Feed CE alone → model predicts y' or y (flipped order)
        delta_prime_t = model.get_margin(ce, y_prime, y)

        # Record this iteration
        history.append({
            "iteration":     i,
            "de":            de,
            "ce":            ce,
            "delta_t":       round(delta_t, 4),
            "delta_prime_t": round(delta_prime_t, 4),
            "gap_de":        round(tau - delta_t, 4),
            "gap_ce":        round(tau - delta_prime_t, 4),
            "converged":     False,
        })

        # Step 3: Convergence check
        # Both margins must reach τ = 0.95
        if delta_t >= tau and delta_prime_t >= tau:
            history[-1]["converged"] = True
            converged = True
            break

        # Step 4: Critique + refine using GAP-informed prompts
        # Refine DE — tell model exactly how far it is from faithful
        de_critique_prompt = get_critique_prompt_phase2(
            sentence0, sentence1,
            de, ce,
            y, y_prime,
            delta_t, delta_prime_t,
            tau
        )
        de_raw = model.get_generated(
            de_critique_prompt, do_sample=False, max_new_tokens=200
        )[0]
        de = parse_explanation(de_raw)

        # Refine CE — uses CE's own margin gap
        ce_refine_prompt = get_ce_refinement_prompt(
            sentence0, sentence1,
            de, ce,
            y, y_prime,
            delta_t, delta_prime_t,
            tau
        )
        ce_raw = model.get_generated(
            ce_refine_prompt, do_sample=False, max_new_tokens=200
        )[0]
        ce = parse_explanation(ce_raw)

    # Final margin check after last iteration (if not converged)
    if not converged:
        delta_t       = model.get_margin(de, y, y_prime)
        delta_prime_t = model.get_margin(ce, y_prime, y)
        history.append({
            "iteration":     max_iters,
            "de":            de,
            "ce":            ce,
            "delta_t":       round(delta_t, 4),
            "delta_prime_t": round(delta_prime_t, 4),
            "gap_de":        round(tau - delta_t, 4),
            "gap_ce":        round(tau - delta_prime_t, 4),
            "converged":     delta_t >= tau and delta_prime_t >= tau,
        })
        if delta_t >= tau and delta_prime_t >= tau:
            converged = True

    item["pcr_phase2"] = {
        "de_original":      item["explanation"]["final"],  # SR-NLE explanation (never changes)
        "de_final":         de,                             # Phase 2 refined explanation
        "ce_final":         ce,                             # Counterfactual explanation
        "y":                y,
        "y_prime":          y_prime,
        "converged":        converged,
        "iterations":       len(history),
        "final_delta_t":    history[-1]["delta_t"],
        "final_delta_prime_t": history[-1]["delta_prime_t"],
        "tau":              tau,
        "history":          history,
    }

    return item


# ------------------------------------------------------------------ #
#  Runner                                                             #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",   type=str,   default="qwen",
                        choices=["llama", "mistral", "qwen", "falcon"])
    parser.add_argument("--dataset_name", type=str,   default="comve")
    parser.add_argument("--prompt_type",  type=str,   default="zs")
    parser.add_argument("--max_iters",    type=int,   default=3)
    parser.add_argument("--tau",          type=float, default=0.95)
    args = parser.parse_args()

    # Paths — reads from SR-NLE directly (NOT Phase 1 output)
    base        = "experiments"
    input_path  = (f"{base}/counterfactual/"
                   f"{args.prompt_type}-{args.dataset_name}-{args.model_name}/"
                   f"explanation_gd.json")
    output_dir  = (f"PCR/experiments/counterfactual/"
                   f"{args.prompt_type}-{args.dataset_name}-{args.model_name}")
    output_path = f"{output_dir}/pcr_phase2.json"

    os.makedirs(output_dir, exist_ok=True)

    # Load SR-NLE explanations
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {input_path}")
    print(f"Model: {args.model_name} | Max iters: {args.max_iters} | tau: {args.tau}")

    # Load model
    model = GenerationModel(args.model_name)

    # Run PCR Phase 2
    results = []
    for item in tqdm(data, desc="PCR Phase 2"):
        item = pcr_phase2(item, model, max_iters=args.max_iters, tau=args.tau)
        results.append(item)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} items to {output_path}")

    # Quick stats
    converged     = sum(1 for r in results if r["pcr_phase2"]["converged"])
    avg_iters     = sum(r["pcr_phase2"]["iterations"] for r in results) / len(results)
    avg_delta_t   = sum(r["pcr_phase2"]["final_delta_t"] for r in results) / len(results)
    avg_delta_pt  = sum(r["pcr_phase2"]["final_delta_prime_t"] for r in results) / len(results)

    print(f"Converged (both margins >= {args.tau}): {converged}/{len(results)}")
    print(f"Average iterations    : {avg_iters:.2f}")
    print(f"Average final Δt      : {avg_delta_t:.4f}")
    print(f"Average final Δ't     : {avg_delta_pt:.4f}")


if __name__ == "__main__":
    main()
