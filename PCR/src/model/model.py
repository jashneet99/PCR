import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import math
from vllm import LLM, SamplingParams

LLM_MODELS = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "falcon":  "tiiuae/Falcon3-7B-Instruct"
}


class GenerationModel:
    """
    Extended version of SR-NLE's GenerationModel — vLLM backend.
    Adds get_margin() for PCR's probabilistic faithfulness check.

    New method:
        get_margin(explanation, y, y_prime)
            → feeds explanation ALONE to the model (no original input x)
            → extracts logprobs for y and y_prime answer tokens
            → applies binary softmax → returns Δt = P(y|e) - P(y'|e)
    """

    def __init__(self, model_name):
        self.model_id = LLM_MODELS[model_name]
        self.system_prompt = "You are a helpful assistant!"

        self.llm = LLM(
            model=self.model_id,
            dtype="bfloat16",
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85")),
            tensor_parallel_size=1,
        )
        self.tokenizer = self.llm.get_tokenizer()

    # ------------------------------------------------------------------ #
    #  Core helpers                                                        #
    # ------------------------------------------------------------------ #

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def get_chat_prompt(self, prompt):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]

    def get_formatted_prompt(self, prompt):
        chat_prompt = self.get_chat_prompt(prompt)
        return self.tokenizer.apply_chat_template(
            chat_prompt,
            tokenize=False,
            add_generation_prompt=True
        )

    def _make_sampling_params(self, **generation_args):
        do_sample  = generation_args.get("do_sample", False)
        max_tokens = generation_args.get("max_new_tokens", 512)
        n          = generation_args.get("num_return_sequences", 1)

        if not do_sample:
            temperature, top_p, top_k = 0.0, 1.0, -1
        else:
            temperature = generation_args.get("temperature") or 1.0
            top_p       = generation_args.get("top_p") or 1.0
            top_k       = generation_args.get("top_k") or -1

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            n=n,
        )

    # ------------------------------------------------------------------ #
    #  SR-NLE generation methods                                           #
    # ------------------------------------------------------------------ #

    def get_generated_batch(self, prompts, **generation_args):
        """Generate outputs for a list of prompts in one batched vLLM call."""
        sampling_params   = self._make_sampling_params(**generation_args)
        formatted_prompts = [self.get_formatted_prompt(p) for p in prompts]
        outputs           = self.llm.generate(formatted_prompts, sampling_params)
        return [[completion.text for completion in req.outputs] for req in outputs]

    def get_generated(self, prompt, **generation_args):
        """Generate outputs for a single prompt."""
        return self.get_generated_batch([prompt], **generation_args)[0]

    def get_messages_generated(self, messages, **generation_args):
        """
        Generate from a full chat messages list (used in PCR dialog loop).
        Preserves full conversation history.
        """
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = self._make_sampling_params(**generation_args)
        outputs         = self.llm.generate([formatted_prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    # ------------------------------------------------------------------ #
    #  PCR Extension — get_margin()                                        #
    # ------------------------------------------------------------------ #

    def get_answer_token_id(self, answer_label: str) -> int:
        """
        Returns the first token ID of the answer label string.
        e.g. "A" → token_id for "A"
             "Sentence 0" → token_id for "Sentence"
        Used to extract logprobs at the answer position.
        """
        tokens = self.tokenizer.encode(
            " " + answer_label,
            add_special_tokens=False
        )
        return tokens[0]

    def get_margin(self, explanation: str, y: str, y_prime: str) -> float:
        """
        PCR's core faithfulness measurement.

        Feeds the explanation ALONE to the model (no original input x).
        Asks the model to predict the answer, then measures:

            Δt = P(y | explanation) - P(y' | explanation)

        Interpretation:
            Δt = +1.0  → explanation perfectly predicts y  (fully faithful)
            Δt =  0.0  → explanation is ambiguous
            Δt = -1.0  → explanation pushes towards wrong prediction

        Args:
            explanation : the explanation text (DE or CE) — no original input
            y           : the factual answer label  (e.g. "A", "Sentence 1")
            y_prime     : the alternative answer label

        Returns:
            Δt (float) in range [-1, +1]
        """
        # Step 1 — Build base prompt with ONLY the explanation
        base_prompt = (
            f"Based only on the following explanation, predict the answer.\n\n"
            f"Explanation: {explanation}\n\n"
            f"Answer (choose one):"
        )

        # Step 2 — Get token IDs and their decoded strings
        y_token_id       = self.get_answer_token_id(y)
        y_prime_token_id = self.get_answer_token_id(y_prime)

        y_token_str       = self.tokenizer.decode([y_token_id])
        y_prime_token_str = self.tokenizer.decode([y_prime_token_id])

        # Step 3 — Append each answer token to the prompt and batch both.
        # prompt_logprobs always includes the actual token's logprob,
        # avoiding vLLM v1's logprobs cap of 20.
        prompt_y       = base_prompt + y_token_str
        prompt_y_prime = base_prompt + y_prime_token_str

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
        )
        outputs = self.llm.generate([prompt_y, prompt_y_prime], sampling_params)

        # Step 4 — Extract log probabilities from the last prompt token position 
        log_y       = outputs[0].prompt_logprobs[-1][y_token_id].logprob
        log_y_prime = outputs[1].prompt_logprobs[-1][y_prime_token_id].logprob

        # Step 5 — Binary softmax → margin Δt
        log_sum   = math.log(math.exp(log_y) + math.exp(log_y_prime))
        p_y       = math.exp(log_y       - log_sum)
        p_y_prime = math.exp(log_y_prime - log_sum)

        return p_y - p_y_prime
