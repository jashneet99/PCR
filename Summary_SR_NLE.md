Yes. I went through the SR-NLE paper: [arXiv:2505.22823](https://arxiv.org/abs/2505.22823), PDF [here](https://arxiv.org/pdf/2505.22823). It is the EMNLP 2025 paper **“Self-Critique and Refinement for Faithful Natural Language Explanations”** by Yingming Wang and Pepa Atanasova.

**Core Idea**
SR-NLE asks: can an LLM improve the **faithfulness** of its own post-hoc explanations without training, external labels, or another model?

The answer in the paper is: yes, if the model is guided with the right feedback.

The pipeline is:

```text
Input x
→ model predicts answer y
→ model generates initial explanation e0
→ feedback is generated
→ explanation is refined
→ repeat for K rounds, usually K = 3
```

**Problem**
LLMs often give explanations that sound reasonable but do not reflect the actual reason behind their prediction. The paper focuses on **post-hoc NLEs**, meaning the explanation is generated after the answer, not as part of the reasoning process.

Example idea:

```text
Model answer changes because of one inserted word.
But the explanation does not mention that word.
So the explanation is likely unfaithful.
```

**How Faithfulness Is Measured**
They use a counterfactual test:

1. Take an original input.
2. Insert one meaningful word, such as an adjective/adverb.
3. If the model’s answer flips, that word likely influenced the model.
4. Then check whether the generated explanation mentions that intervened word.

Metric:

```text
Unfaithfulness = unfaithful explanations / counter instances
```

Lower is better.

**Feedback Types**
The paper tests two major feedback families:

1. **NLF: Natural Language Feedback**
   The model critiques its own explanation in free text.

2. **IWF: Important Word Feedback**
   The model is told which input words were important to its prediction. Then it refines the explanation using those words.

IWF has three variants:

- `IWF-Pmt`: prompt the LLM to rank important words
- `IWF-Attn`: use attention-based attribution
- `IWF-IG`: use Integrated Gradients

The best overall variant is **IWF-Attn**.

**Datasets and Models**
Datasets:

- ComVE
- ECQA
- e-SNLI

Models:

- Falcon 7B
- Llama 3.1 8B
- Mistral 7B
- Qwen 2.5 7B

They use 1,000 test instances per dataset, and generate counterfactual interventions for evaluation.

**Main Results**
The big headline:

```text
Init-NLE average unfaithfulness: 54.81%
Best SR-NLE average unfaithfulness: 36.02%
Absolute reduction: 18.79 percentage points
```

Table 1 shows:

- Init-NLE: `54.81`
- SC-NLE: `49.74`
- NLF: `47.58`
- IWF-Pmt: `37.23`
- IWF-Attn: `36.02`
- IWF-IG: `36.30`

So important-word feedback is much stronger than natural language self-feedback.

**Most Important Takeaway**
The paper’s strongest insight is:

```text
Natural language feedback helps, but important-word feedback helps much more.
```

Why? Because NLF may produce vague suggestions, while IWF gives the model concrete input tokens that should appear in the explanation.

**Refinement Behavior**
The paper finds:

- Most improvement happens in the first refinement round.
- Rounds 2 and 3 still help, but gains shrink.
- Explanations become longer over rounds.
- Longer explanations tend to be more faithful, but the paper argues the gain is not only because of length. It is because refinements include more relevant reasoning words.

**Limitations**
The paper itself notes several limitations:

- It only studies post-hoc explanations, not chain-of-thought or jointly generated rationales.
- Faithfulness is measured only through counterfactual string matching.
- Attention and Integrated Gradients are imperfect attribution methods.
- Experiments are only on models around 7B-8B scale.

**Connection To Your Repo**
Your repo is basically implementing this paper:

- `answer_runner.py` = answer generation
- `explanation_runner.py` = initial NLE generation
- `feedback_runner.py` = NLF/IWF feedback
- `refinement_runner.py` = iterative refinement
- `counter.py` = filters flipped counterfactuals
- `faithfulness.py` = checks whether `edit_word` appears in explanation

Your **PCR** extension builds on top of SR-NLE by saying: mentioning the edit word is useful, but maybe too shallow. PCR tries to make explanations contrastive and mutually exclusive between actual and alternative answers. That is a very natural next step from this paper’s limitation.


#### PCR AS IMPROVEMENT ON TOP OF SR-NLE 

Yes. PCR is best framed as an improvement **because it fixes the exact weak spot that SR-NLE leaves open**.

**Core Relation**
SR-NLE improves explanations by asking the model to critique or refine them using feedback, especially important-word feedback. But its faithfulness signal is still mostly:

```text
Did the explanation mention the edit_word?
```

That helps, but it can become shallow. The model may simply insert the important word into the explanation without making the explanation truly specific to the selected answer.

PCR changes the objective from:

```text
Mention the important word.
```

to:

```text
Make the explanation uniquely justify this answer and not the opposite answer.
```

That is the main conceptual upgrade.

**Why SR-NLE Plateaus**
In the SR-NLE paper, the best method reduces average unfaithfulness from `54.81%` to `36.02%`, which is strong. But the mechanism still depends on feedback that tells the model what to improve, not whether the explanation logically discriminates between answers.

So SR-NLE can produce explanations like:

```text
Sentence 0 is wrong because grizzly fierce bears hate honey.
```

This mentions `fierce`, so it may pass the metric. But it may still not deeply explain why Sentence 0 is wrong compared with Sentence 1.

That is where PCR enters.

**What PCR Adds**
PCR introduces a counterfactual explanation, CE:

```text
DE = explanation for actual answer y
CE = explanation for opposite answer y'
```

Then PCR asks:

```text
Are DE and CE mutually exclusive?
Can DE prove y while excluding y'?
Can CE prove y' while excluding y?
```

So PCR makes the model reason contrastively. It forces explanations to become answer-specific, not just word-containing.

**Phase 1 Improvement**
PCR Phase 1 uses a prompt-based mutual exclusivity check.

SR-NLE says:

```text
Improve your explanation.
```

PCR Phase 1 says:

```text
Your explanation and the opposite explanation overlap.
Rewrite it so only your answer is justified.
```

This directly targets vague explanations.

**Phase 2 Improvement**
PCR Phase 2 goes further. Instead of trusting the model’s yes/no judgment, it measures whether the explanation alone predicts the answer:

```text
Delta = P(y | explanation) - P(y' | explanation)
```

So Phase 2 adds a quantitative signal. The model gets told how far the explanation is from strongly supporting the intended answer.

That is stronger than SR-NLE’s normal self-feedback.

**How To Present It**
One clean explanation:

```text
SR-NLE improves explanation faithfulness through self-critique, but it mainly encourages the model to include important words. PCR improves on SR-NLE by making explanations contrastive: the refined explanation must not only mention the causal word, but also uniquely justify the chosen answer over the counterfactual answer.
```

**Your Result Story**
In your repo notes, this matches the numbers:

```text
Qwen ComVE:
SR-NLE init       28.09% faith rate
SR-NLE best       36.60%
PCR Phase 1       53.62%
PCR Phase 2       65.53%
```

So PCR is not just an add-on. It is a targeted fix for SR-NLE’s main limitation: SR-NLE improves wording, while PCR improves answer-specific faithfulness.
