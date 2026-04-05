"""
================================================================================
PCR — FINAL RESULTS (Qwen model, ComVE dataset)
SR-NLE vs PCR Phase 1 vs PCR Phase 2
================================================================================

Run this file to print the full results:
    python PCR/pcr_final_results.py
"""


def print_results():
    print("""
================================================================================
FINAL RESULTS — ALL STAGES
Qwen model | ComVE dataset | 235 counterfactual items
================================================================================

  Stage                    Faithful    Unfaithful   Faith Rate   Unfaith Rate
  ──────────────────────────────────────────────────────────────────────────
  SR-NLE init              66/235      169/235       28.09%       71.91%
  SR-NLE iter 0            78/235      157/235       33.19%       66.81%
  SR-NLE iter 1            85/235      150/235       36.17%       63.83%
  SR-NLE iter 2 (best)     86/235      149/235       36.60%       63.40%
  ──────────────────────────────────────────────────────────────────────────
  PCR Phase 1             126/235      109/235       53.62%       46.38%
  PCR Phase 2             154/235       81/235       65.53%       34.47%
  ──────────────────────────────────────────────────────────────────────────

================================================================================
IMPROVEMENT OVER SR-NLE BEST (iter 2)
================================================================================

  PCR Phase 1 vs SR-NLE best  →  +17.02%  faith rate improvement
  PCR Phase 2 vs SR-NLE best  →  +28.93%  faith rate improvement
  PCR Phase 2 vs PCR Phase 1  →  +11.91%  additional improvement

================================================================================
IMPROVEMENT OVER SR-NLE INIT (baseline)
================================================================================

  PCR Phase 1 vs SR-NLE init  →  +25.53%
  PCR Phase 2 vs SR-NLE init  →  +37.44%

================================================================================
PCR PROCESS STATS
================================================================================

  Phase 1:
    Converged (ME=YES)    : 135/235  (57.4%)
    Average iterations    : 1.47
    Regressions           : 0

  Phase 2:
    Converged (Δt>=0.95)  : 0/235    (0.0%)  ← τ=0.95 is very strict
    Average iterations    : 4.00
    Average final Δt      : 0.1415
    Average final Δ't     : -0.1334

================================================================================
HOW TO EXPLAIN TO YOUR SENIOR
================================================================================

  ONE LINE:
    "We improved unfaithfulness from 63.40% (SR-NLE best) down to 34.47%
     (PCR Phase 2) — a 28.93 percentage point reduction."

  FULL:
    "SR-NLE's best result after 3 refinement iterations was 36.60% faith rate.
     PCR Phase 1, using prompt-only mutual exclusivity refinement, improved
     this to 53.62%. PCR Phase 2, using probability margin guidance (Δt),
     improved it further to 65.53% — all without changing the model or data."

  ON PHASE 2 CONVERGENCE = 0:
    "Phase 2 never formally converged because the threshold τ=0.95 is very
     strict and average Δt only reached 0.14. However, the probability-guided
     critique still produced significantly better explanations, achieving the
     highest faith rate of 65.53%. This suggests DE refinement benefits
     strongly from quantitative margin signals even without full convergence."

================================================================================
SAVED FILES
================================================================================

  Phase 1 output    : PCR/experiments/counterfactual/zs-comve-qwen/pcr_phase1.json
  Phase 2 output    : PCR/experiments/counterfactual/zs-comve-qwen/pcr_phase2.json
  Results CSV       : PCR/logs/df/pcr_faithfulness_comparison.csv
  SR-NLE results    : logs/df/faithfulness_results.csv
""")


if __name__ == "__main__":
    print_results()
