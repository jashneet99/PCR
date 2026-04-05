"""
PCR Prompt Templates for ComVE dataset.

ComVE task: Given two sentences, identify which one violates commonsense.
Answer options: (A) = Sentence 0, (B) = Sentence 1
"""


def get_ce_generation_prompt(sentence0: str, sentence1: str, y: str, y_prime: str) -> str:
    """
    Generate counterfactual explanation (CE).
    Given the same input, ask the model to explain from the OPPOSITE prediction's perspective.
    """
    return (
        f"You are given two sentences. One violates commonsense.\n\n"
        f"Sentence 0: {sentence0}\n"
        f"Sentence 1: {sentence1}\n\n"
        f"The actual answer is ({y}) — that sentence violates commonsense.\n\n"
        f"Now imagine the answer were ({y_prime}) instead.\n"
        f"Explain in one concise sentence why ({y_prime}) might violate commonsense "
        f"based on the same sentences.\n\n"
        f"You must give your explanation only in the following format:\n"
        f"Explanation: [your explanation here.]"
    )


def get_mutual_exclusivity_prompt(
    sentence0: str, sentence1: str,
    de: str, ce: str,
    y: str, y_prime: str
) -> str:
    """
    Phase 1 — Prompt-only check.
    Ask model: are DE and CE mutually exclusive?
    If Yes → DE is specific enough → faithful
    If No  → DE is vague/overlapping → needs refinement
    """
    return (
        f"You are given two sentences and two explanations.\n\n"
        f"Sentence 0: {sentence0}\n"
        f"Sentence 1: {sentence1}\n\n"
        f"Explanation for answer ({y}): {de}\n"
        f"Explanation for answer ({y_prime}): {ce}\n\n"
        f"Are these two explanations mutually exclusive?\n"
        f"Mutually exclusive means: reading Explanation 1 alone, "
        f"you can clearly tell the answer is ({y}) and NOT ({y_prime}), "
        f"and vice versa for Explanation 2.\n\n"
        f"Answer with only YES or NO."
    )


def get_critique_prompt_phase1(
    sentence0: str, sentence1: str,
    de: str, ce: str,
    y: str, y_prime: str
) -> str:
    """
    Phase 1 critique — no probabilities, just asks model to improve.
    """
    return (
        f"You are given two sentences and two explanations that are NOT mutually exclusive.\n\n"
        f"Sentence 0: {sentence0}\n"
        f"Sentence 1: {sentence1}\n\n"
        f"Current explanation for ({y}): {de}\n"
        f"Current explanation for ({y_prime}): {ce}\n\n"
        f"The explanations are overlapping — reading either one does not clearly "
        f"distinguish which sentence violates commonsense.\n\n"
        f"Rewrite the explanation for ({y}) so it:\n"
        f"  1. Clearly proves why ({y}) is the answer\n"
        f"  2. Explicitly excludes the reasoning of ({y_prime})\n\n"
        f"You must give your explanation only in the following format:\n"
        f"Explanation: [your explanation here.]"
    )


def get_critique_prompt_phase2(
    sentence0: str, sentence1: str,
    de: str, ce: str,
    y: str, y_prime: str,
    delta_t: float, delta_prime_t: float,
    tau: float
) -> str:
    """
    Phase 2 critique — uses actual margin values (Δt and Δ't) in the prompt.
    Tells the model exactly HOW FAR each explanation is from faithful.
    """
    gap_de = tau - delta_t
    gap_ce = tau - delta_prime_t

    return (
        f"You are given two sentences and two explanations that need improvement.\n\n"
        f"Sentence 0: {sentence0}\n"
        f"Sentence 1: {sentence1}\n\n"
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
        f"precisely proves why ({y}) violates commonsense "
        f"and explicitly excludes the reasoning of ({y_prime}).\n\n"
        f"You must give your explanation only in the following format:\n"
        f"Explanation: [your explanation here.]"
    )


def get_ce_refinement_prompt(
    sentence0: str, sentence1: str,
    de: str, ce: str,
    y: str, y_prime: str,
    delta_t: float, delta_prime_t: float,
    tau: float
) -> str:
    """
    Refine the counterfactual explanation (CE) using margin feedback.
    """
    gap_ce = tau - delta_prime_t

    return (
        f"You are given two sentences and need to improve a counterfactual explanation.\n\n"
        f"Sentence 0: {sentence0}\n"
        f"Sentence 1: {sentence1}\n\n"
        f"Explanation for ({y}): {de}\n"
        f"Current explanation for ({y_prime}): {ce}\n\n"
        f"The explanation for ({y_prime}) has margin = {delta_prime_t:.3f} "
        f"(needs {gap_ce:.3f} more to reach threshold {tau}).\n\n"
        f"Rewrite the explanation for ({y_prime}) so it clearly proves "
        f"why ({y_prime}) would violate commonsense and explicitly "
        f"excludes the reasoning of ({y}).\n\n"
        f"You must give your explanation only in the following format:\n"
        f"Explanation: [your explanation here.]"
    )
