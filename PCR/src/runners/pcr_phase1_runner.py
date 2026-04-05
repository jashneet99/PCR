"""
PCR Phase 1 Runner — Prompt-only PCR
=====================================
Uses SR-NLE's explanation as the starting DE.
Generates CE (counterfactual explanation).
Asks model: "Are DE and CE mutually exclusive?" Yes/No
If No → critique → refine both → repeat (max_iters)
If Yes → explanation is faithful → stop

Input  : ../../experiments/counterfactual/zs-comve-{model}/explanation_gd.json
Output : ../../experiments/counterfactual/zs-comve-{model}/pcr_phase1.json

Run from SR-NLE/ root:
    python PCR/src/runners/pcr_phase1_runner.py \
        --model_name qwen \
        --max_iters 3
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
    get_mutual_exclusivity_prompt,
    get_critique_prompt_phase1,
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


def parse_yes_no(text: str) -> bool:
    """Returns True if model answered YES, False otherwise."""
    text = text.strip().upper()
    return text.startswith("YES")


def answer_to_labels(answer_final: str):
    """
    Convert answer final (e.g. 'A' or 'B') to
    y (what model predicted) and y_prime (the other option).
    """
    if answer_final.strip().upper() == "A":
        return "A", "B"
    else:
        return "B", "A"


# ------------------------------------------------------------------ #
#  Main PCR Phase 1 Loop                                              #
# ------------------------------------------------------------------ #

def pcr_phase1(item: dict, model: GenerationModel, max_iters: int = 3) -> dict:
    """
    Runs PCR Phase 1 on a single item.

    Steps:
        1. Take SR-NLE explanation as DE
        2. Generate CE (opposite explanation)
        3. Check mutual exclusivity (Yes/No)
        4. If No → critique → refine DE + CE → repeat
        5. Return updated item with PCR results
    """
    sentence0   = item["sentence0"]
    sentence1   = item["sentence1"]
    # edit_word   = item["edit_word"]
    de          = item["explanation"]["final"]
    y, y_prime  = answer_to_labels(item["answer"]["final"])

    # --- Step 1: Generate CE ---
    ce_prompt = get_ce_generation_prompt(sentence0, sentence1, y, y_prime)
    ce_raw    = model.get_generated(ce_prompt, do_sample=False, max_new_tokens=150)[0]
    ce        = parse_explanation(ce_raw)

    history = [{
        "iteration": -1,
        "de": de,
        "ce": ce,
        "mutually_exclusive": None,
    }]

    converged = False

    # --- Steps 3-4: Iterative refinement ---
    for i in range(max_iters):
        # Check mutual exclusivity
        me_prompt  = get_mutual_exclusivity_prompt(sentence0, sentence1, de, ce, y, y_prime)
        me_raw     = model.get_generated(me_prompt, do_sample=False, max_new_tokens=10)[0]
        is_me      = parse_yes_no(me_raw)

        history[-1]["mutually_exclusive"] = is_me
        history[-1]["me_response"]        = me_raw.strip()

        if is_me:
            converged = True
            break

        # Critique + refine DE
        de_critique_prompt = get_critique_prompt_phase1(
            sentence0, sentence1, de, ce, y, y_prime
        )
        de_raw = model.get_generated(
            de_critique_prompt, do_sample=False, max_new_tokens=150
        )[0]
        de = parse_explanation(de_raw)

        # Refine CE (ask to improve from y_prime's perspective)
        ce_critique_prompt = get_critique_prompt_phase1(
            sentence0, sentence1, ce, de, y_prime, y  # swapped roles
        )
        ce_raw = model.get_generated(
            ce_critique_prompt, do_sample=False, max_new_tokens=150
        )[0]
        ce = parse_explanation(ce_raw)

        history.append({
            "iteration":          i,
            "de":                 de,
            "ce":                 ce,
            "mutually_exclusive": None,
        })

    # Final mutual exclusivity check
    if not converged:
        me_prompt = get_mutual_exclusivity_prompt(sentence0, sentence1, de, ce, y, y_prime)
        me_raw    = model.get_generated(me_prompt, do_sample=False, max_new_tokens=10)[0]
        is_me     = parse_yes_no(me_raw)
        history[-1]["mutually_exclusive"] = is_me
        history[-1]["me_response"]        = me_raw.strip()

    item["pcr_phase1"] = {
        "de_original":  item["explanation"]["final"],   # SR-NLE's original explanation
        "de_final":     de,                              # PCR's refined explanation
        "ce_final":     ce,                              # Counterfactual explanation
        "y":            y,
        "y_prime":      y_prime,
        "converged":    converged,
        "iterations":   len(history) - 1,
        "history":      history,
    }

    return item


# ------------------------------------------------------------------ #
#  Runner                                                             #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen",
                        choices=["llama", "mistral", "qwen", "falcon"])
    parser.add_argument("--dataset_name", type=str, default="comve")
    parser.add_argument("--prompt_type", type=str, default="zs")
    parser.add_argument("--max_iters", type=int, default=3)
    args = parser.parse_args()

    # Paths
    base          = "experiments"
    input_path    = (f"{base}/counterfactual/"
                     f"{args.prompt_type}-{args.dataset_name}-{args.model_name}/"
                     f"explanation_gd.json")
    output_dir    = (f"PCR/experiments/counterfactual/"
                     f"{args.prompt_type}-{args.dataset_name}-{args.model_name}")
    output_path   = f"{output_dir}/pcr_phase1.json"

    os.makedirs(output_dir, exist_ok=True)

    # Load SR-NLE explanations
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {input_path}")
    print(f"Model: {args.model_name} | Max iters: {args.max_iters}")

    # Load model
    model = GenerationModel(args.model_name)

    # Run PCR Phase 1
    results = []
    for item in tqdm(data, desc="PCR Phase 1"):
        item = pcr_phase1(item, model, max_iters=args.max_iters)
        results.append(item)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} items to {output_path}")

    # Quick stats
    converged    = sum(1 for r in results if r["pcr_phase1"]["converged"])
    avg_iters    = sum(r["pcr_phase1"]["iterations"] for r in results) / len(results)
    print(f"Converged (mutually exclusive): {converged}/{len(results)}")
    print(f"Average iterations: {avg_iters:.2f}")


if __name__ == "__main__":
    main()
