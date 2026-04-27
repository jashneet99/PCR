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


def load_prompts(dataset_name: str):
    if dataset_name == "ecqa":
        from prompts.ecqa_prompts import (
            get_ce_generation_prompt,
            get_mutual_exclusivity_prompt,
            get_critique_prompt_phase1,
        )
    elif dataset_name == "esnli":
        from prompts.esnli_prompts import (
            get_ce_generation_prompt,
            get_mutual_exclusivity_prompt,
            get_critique_prompt_phase1,
        )
    else:
        from prompts.comve_prompts import (
            get_ce_generation_prompt,
            get_mutual_exclusivity_prompt,
            get_critique_prompt_phase1,
        )
    return get_ce_generation_prompt, get_mutual_exclusivity_prompt, get_critique_prompt_phase1


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


def answer_to_labels(answer_final: str, original_answer: str = None):
    """
    Convert answer final to y and y_prime.
    ComVE: only A/B, so y_prime is trivially the other one.
    ECQA:  A-E, so y_prime must be supplied (original answer before the flip).
    """
    y = answer_final.strip().upper()
    if original_answer is not None:
        y_prime = original_answer.strip().upper()
    else:
        y_prime = "B" if y == "A" else "A"
    return y, y_prime


# ------------------------------------------------------------------ #
#  Main PCR Phase 1 Loop                                              #
# ------------------------------------------------------------------ #

def pcr_phase1(item: dict, model: GenerationModel, max_iters: int = 3,
               dataset_name: str = "comve", original_answer: str = None) -> dict:
    """
    Runs PCR Phase 1 on a single item.

    Steps:
        1. Take SR-NLE explanation as DE
        2. Generate CE (opposite explanation)
        3. Check mutual exclusivity (Yes/No)
        4. If No → critique → refine DE + CE → repeat
        5. Return updated item with PCR results
    """
    de         = item["explanation"]["final"]
    y, y_prime = answer_to_labels(item["answer"]["final"], original_answer)

    if dataset_name == "ecqa":
        question = item["question"]
        choices  = item["choices"]
        get_ce_generation_prompt, get_mutual_exclusivity_prompt, get_critique_prompt_phase1 = load_prompts("ecqa")
        ce_prompt = get_ce_generation_prompt(question, choices, y, y_prime)
    elif dataset_name == "esnli":
        premise    = item["premise"]
        hypothesis = item["hypothesis"]
        choices    = item["choices"]
        get_ce_generation_prompt, get_mutual_exclusivity_prompt, get_critique_prompt_phase1 = load_prompts("esnli")
        ce_prompt = get_ce_generation_prompt(premise, hypothesis, choices, y, y_prime)
    else:
        sentence0 = item["sentence0"]
        sentence1 = item["sentence1"]
        get_ce_generation_prompt, get_mutual_exclusivity_prompt, get_critique_prompt_phase1 = load_prompts("comve")
        ce_prompt = get_ce_generation_prompt(sentence0, sentence1, y, y_prime)

    ce_raw = model.get_generated(ce_prompt, do_sample=False, max_new_tokens=150)[0]
    ce     = parse_explanation(ce_raw)

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
        if dataset_name == "ecqa":
            me_prompt = get_mutual_exclusivity_prompt(question, choices, de, ce, y, y_prime)
        elif dataset_name == "esnli":
            me_prompt = get_mutual_exclusivity_prompt(premise, hypothesis, choices, de, ce, y, y_prime)
        else:
            me_prompt = get_mutual_exclusivity_prompt(sentence0, sentence1, de, ce, y, y_prime)
        me_raw = model.get_generated(me_prompt, do_sample=False, max_new_tokens=10)[0]
        is_me  = parse_yes_no(me_raw)

        history[-1]["mutually_exclusive"] = is_me
        history[-1]["me_response"]        = me_raw.strip()

        if is_me:
            converged = True
            break

        # Critique + refine DE
        if dataset_name == "ecqa":
            de_critique_prompt = get_critique_prompt_phase1(question, choices, de, ce, y, y_prime)
            ce_critique_prompt = get_critique_prompt_phase1(question, choices, ce, de, y_prime, y)
        elif dataset_name == "esnli":
            de_critique_prompt = get_critique_prompt_phase1(premise, hypothesis, choices, de, ce, y, y_prime)
            ce_critique_prompt = get_critique_prompt_phase1(premise, hypothesis, choices, ce, de, y_prime, y)
        else:
            de_critique_prompt = get_critique_prompt_phase1(sentence0, sentence1, de, ce, y, y_prime)
            ce_critique_prompt = get_critique_prompt_phase1(sentence0, sentence1, ce, de, y_prime, y)

        de_raw = model.get_generated(de_critique_prompt, do_sample=False, max_new_tokens=150)[0]
        de     = parse_explanation(de_raw)
        ce_raw = model.get_generated(ce_critique_prompt, do_sample=False, max_new_tokens=150)[0]
        ce     = parse_explanation(ce_raw)

        history.append({
            "iteration":          i,
            "de":                 de,
            "ce":                 ce,
            "mutually_exclusive": None,
        })

    # Final mutual exclusivity check
    if not converged:
        if dataset_name == "ecqa":
            me_prompt = get_mutual_exclusivity_prompt(question, choices, de, ce, y, y_prime)
        elif dataset_name == "esnli":
            me_prompt = get_mutual_exclusivity_prompt(premise, hypothesis, choices, de, ce, y, y_prime)
        else:
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

    # For ECQA/eSNLI: load original answers to determine y_prime
    original_answers = {}
    if args.dataset_name in ("ecqa", "esnli"):
        org_path = (f"{base}/original/{args.prompt_type}-{args.dataset_name}-{args.model_name}"
                    f"/answer_gd.json")
        with open(org_path, "r", encoding="utf-8") as f:
            org_data = json.load(f)
        original_answers = {item["idx"]: item["answer"]["final"] for item in org_data}

    # Load model
    model = GenerationModel(args.model_name)

    # Resume from existing output if available
    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} already processed items")
        data = data[len(results):]

    # Run PCR Phase 1
    for item in tqdm(data, desc="PCR Phase 1"):
        # Skip items with no valid explanation (None answer from model)
        if item.get("explanation") is None or item["explanation"].get("final") is None:
            item["pcr_phase1"] = None
            results.append(item)
            continue
        orig_ans = original_answers.get(item["idx"]) if args.dataset_name == "ecqa" else None
        item = pcr_phase1(item, model, max_iters=args.max_iters,
                          dataset_name=args.dataset_name, original_answer=orig_ans)
        results.append(item)
        # Save incrementally after each item
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} items to {output_path}")

    # Quick stats
    valid_results = [r for r in results if r["pcr_phase1"] is not None]
    converged     = sum(1 for r in valid_results if r["pcr_phase1"]["converged"])
    avg_iters     = sum(r["pcr_phase1"]["iterations"] for r in valid_results) / len(valid_results) if valid_results else 0
    print(f"Converged (mutually exclusive): {converged}/{len(valid_results)} (skipped {len(results) - len(valid_results)} None items)")
    print(f"Average iterations: {avg_iters:.2f}")


if __name__ == "__main__":
    main()
