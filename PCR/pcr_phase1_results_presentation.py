"""
================================================================================
PCR — RESULTS PRESENTATION (Phase 1 + Phase 2)
How to explain the results to your senior
================================================================================

Run this file to print the presentation:
    python PCR/pcr_phase1_results_presentation.py
"""


def print_presentation():
    print("""
================================================================================
PCR — FINAL RESULTS (Qwen model, ComVE dataset)
================================================================================

FULL RESULTS TABLE
================================================================================

    Stage                   Faithful    Unfaithful   Faith Rate   Unfaith Rate
    ──────────────────────────────────────────────────────────────────────────
    SR-NLE init             66/235      169/235       28.09%       71.91%
    SR-NLE iter 0           78/235      157/235       33.19%       66.81%
    SR-NLE iter 1           85/235      150/235       36.17%       63.83%
    SR-NLE iter 2 (best)    86/235      149/235       36.60%       63.40%
    ──────────────────────────────────────────────────────────────────────────
    PCR Phase 1            126/235      109/235       53.62%       46.38%
    PCR Phase 2            154/235       81/235       65.53%       34.47%
    ──────────────────────────────────────────────────────────────────────────

CORE METRIC: UNFAITHFUL RATE (lower is better)
------------------------------------------------
    SR-NLE best (iter 2)  →  63.40%  unfaithful  (149/235 not faithful)
    PCR Phase 1           →  46.38%  unfaithful  (109/235 not faithful)
    PCR Phase 2           →  34.47%  unfaithful  ( 81/235 not faithful)

FAITH RATE (higher is better)
------------------------------
    SR-NLE best (iter 2)  →  36.60%  faithful    ( 86/235 faithful)
    PCR Phase 1           →  53.62%  faithful    (126/235 faithful)
    PCR Phase 2           →  65.53%  faithful    (154/235 faithful)

IMPROVEMENT OVER SR-NLE BEST
------------------------------
    PCR Phase 1  →  +17.02%
    PCR Phase 2  →  +28.93%
    Phase 2 over Phase 1  →  +11.91%

NOTE: Faith rate and unfaith rate are two sides of the same coin.
      Both show the same improvement — just in opposite directions.

================================================================================
HOW TO EXPLAIN TO YOUR SENIOR
================================================================================

ONE LINE:
---------
    "We improved unfaithfulness from 63.40% (SR-NLE best) down to 34.47%
     (PCR Phase 2) — a 28.93 percentage point reduction."

FULL EXPLANATION:
-----------------
    "SR-NLE's best result after 3 refinement iterations was 36.60% faith rate.
     PCR Phase 1, using prompt-only mutual exclusivity refinement, improved
     this to 53.62%. PCR Phase 2, using probability margin guidance (Δt),
     improved it further to 65.53% — all without changing the model or data."

IF SENIOR ASKS WHAT UNFAITHFUL MEANS:
--------------------------------------
    "Unfaithful means the explanation the model gives does not mention
     the key word that actually caused it to make that decision.
     So the model is giving a reason that is not its real reason."

IF SENIOR ASKS WHY PHASE 1 IMPROVED:
--------------------------------------
    "SR-NLE told the model to mention a word.
     PCR Phase 1 told the model to make its explanation exclusively prove
     its answer. That deeper instruction naturally produced better,
     more specific explanations."

IF SENIOR ASKS WHY PHASE 2 IMPROVED FURTHER:
----------------------------------------------
    "Phase 1 used the model's own YES/NO judgment — subjective.
     Phase 2 measured faithfulness mathematically using probability margins.
     It told the model exactly how far its explanation was from faithful
     using a number (GAP), not just words. That quantitative signal
     produced even more precise explanations."

IF SENIOR ASKS WHY PHASE 2 CONVERGED = 0:
-------------------------------------------
    "The threshold τ=0.95 is very strict — the explanation must almost
     certainly predict the correct answer on its own. Average Δt only
     reached 0.14. But convergence and faith rate are different things.
     The probability-guided critique still rewrote explanations more
     precisely in each iteration, achieving 65.53% faith rate — the
     best result across all methods."

================================================================================
PCR PROCESS STATS
================================================================================

    Phase 1:
        Converged (ME=YES)   : 135/235  (57.4%)
        Average iterations   : 1.47
        Regressions          : 0

    Phase 2:
        Converged (Δt>=0.95) : 0/235    (0.0%)
        Average iterations   : 4.00
        Average final Δt     : 0.1415
        Average final Δ't    : -0.1334

================================================================================
SAVED FILES
================================================================================

    Phase 1 output  : PCR/experiments/counterfactual/zs-comve-qwen/pcr_phase1.json
    Phase 2 output  : PCR/experiments/counterfactual/zs-comve-qwen/pcr_phase2.json
    Results CSV     : PCR/logs/df/pcr_faithfulness_comparison.csv
    SR-NLE results  : logs/df/faithfulness_results.csv
""")


if __name__ == "__main__":
    print_presentation()
