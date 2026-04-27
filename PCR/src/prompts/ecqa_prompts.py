"""
PCR Prompt Templates for ECQA dataset.

ECQA task: 5-choice commonsense QA.
Answer options: (A), (B), (C), (D), (E)
"""


def format_choices(choices: list) -> str:
    """Format choices list as (A) choice1 (B) choice2 ..."""
    labels = ["A", "B", "C", "D", "E"]
    return "\n".join(f"({labels[i]}) {c.capitalize()}" for i, c in enumerate(choices))


def get_ce_generation_prompt(question: str, choices: list, y: str, y_prime: str) -> str:
    """
    Generate counterfactual explanation (CE).
    Given the same question, ask the model to explain from y_prime's perspective.
    """
    formatted = format_choices(choices)
    return (
        f"You are given a multiple-choice commonsense question.\n\n"
        f"Question: {question}\n"
        f"Answer Options:\n{formatted}\n\n"
        f"The actual answer is ({y}).\n\n"
        f"Now imagine the answer were ({y_prime}) instead.\n"
        f"Explain in one concise sentence why ({y_prime}) might be the correct answer "
        f"based on the same question.\n\n"
        f"You must give your explanation only in the following format:\n"
        f"Explanation: [your explanation here.]"
    )


def get_mutual_exclusivity_prompt(
    question: str, choices: list,
    de: str, ce: str,
    y: str, y_prime: str
) -> str:
    """
    Phase 1 — Prompt-only check.
    Ask model: are DE and CE mutually exclusive?
    """
    formatted = format_choices(choices)
    return (
        f"You are given a multiple-choice commonsense question and two explanations.\n\n"
        f"Question: {question}\n"
        f"Answer Options:\n{formatted}\n\n"
        f"Explanation for answer ({y}): {de}\n"
        f"Explanation for answer ({y_prime}): {ce}\n\n"
        f"Are these two explanations mutually exclusive?\n"
        f"Mutually exclusive means: reading Explanation 1 alone, "
        f"you can clearly tell the answer is ({y}) and NOT ({y_prime}), "
        f"and vice versa for Explanation 2.\n\n"
        f"Answer with only YES or NO."
    )


def get_critique_prompt_phase1(
    question: str, choices: list,
    de: str, ce: str,
    y: str, y_prime: str
) -> str:
    """
    Phase 1 critique — no probabilities, just asks model to improve.
    """
    formatted = format_choices(choices)
    return (
        f"You are given a multiple-choice commonsense question and two explanations "
        f"that are NOT mutually exclusive.\n\n"
        f"Question: {question}\n"
        f"Answer Options:\n{formatted}\n\n"
        f"Current explanation for ({y}): {de}\n"
        f"Current explanation for ({y_prime}): {ce}\n\n"
        f"The explanations are overlapping — reading either one does not clearly "
        f"distinguish which answer is correct.\n\n"
        f"Rewrite the explanation for ({y}) so it:\n"
        f"  1. Clearly proves why ({y}) is the correct answer\n"
        f"  2. Explicitly excludes the reasoning of ({y_prime})\n\n"
        f"You must give your explanation only in the following format:\n"
        f"Explanation: [your explanation here.]"
    )


def get_critique_prompt_phase2(
    question: str, choices: list,
    de: str, ce: str,
    y: str, y_prime: str,
    delta_t: float, delta_prime_t: float,
    tau: float
) -> str:
    """
    Phase 2 critique — uses actual margin values (Δt and Δ't) in the prompt.
    """
    gap_de = tau - delta_t
    gap_ce = tau - delta_prime_t
    formatted = format_choices(choices)

    return (
        f"You are given a multiple-choice commonsense question and two explanations "
        f"that need improvement.\n\n"
        f"Question: {question}\n"
        f"Answer Options:\n{formatted}\n\n"
        f"Current explanation for ({y}): {de}\n"
        f"Current explanation for ({y_prime}): {ce}\n\n"
        f"Faithfulness measurement results:\n"
        f"  - Explanation for ({y})   : margin = {delta_t:.3f}  "
        f"(needs {gap_de:.3f} more to reach threshold {tau})\n"
        f"  - Explanation for ({y_prime}): margin = {delta_prime_t:.3f}  "
        f"(needs {gap_ce:.3f} more to reach threshold {tau})\n\n"
        f"Both margins are below the required threshold of {tau}.\n"
        f"This means the explanation for ({y}) is still ambiguous — "
        f"it does not strongly enough prove ({y}) over ({y_prime}).\n\n"
        f"Rewrite the explanation for ({y}) so it more specifically and "
        f"precisely proves why ({y}) is the correct commonsense answer "
        f"and explicitly excludes the reasoning of ({y_prime}).\n\n"
        f"You must give your explanation only in the following format:\n"
        f"Explanation: [your explanation here.]"
    )


def get_ce_refinement_prompt(
    question: str, choices: list,
    de: str, ce: str,
    y: str, y_prime: str,
    delta_t: float, delta_prime_t: float,
    tau: float
) -> str:
    """
    Refine the counterfactual explanation (CE) using margin feedback.
    """
    gap_ce = tau - delta_prime_t
    formatted = format_choices(choices)

    return (
        f"You are given a multiple-choice commonsense question and need to improve "
        f"a counterfactual explanation.\n\n"
        f"Question: {question}\n"
        f"Answer Options:\n{formatted}\n\n"
        f"Explanation for ({y}): {de}\n"
        f"Current explanation for ({y_prime}): {ce}\n\n"
        f"The explanation for ({y_prime}) has margin = {delta_prime_t:.3f} "
        f"(needs {gap_ce:.3f} more to reach threshold {tau}).\n\n"
        f"Rewrite the explanation for ({y_prime}) so it clearly proves "
        f"why ({y_prime}) would be the correct commonsense answer and explicitly "
        f"excludes the reasoning of ({y}).\n\n"
        f"You must give your explanation only in the following format:\n"
        f"Explanation: [your explanation here.]"
    )
