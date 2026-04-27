import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from vllm import LLM, SamplingParams

LLM_MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "falcon": "tiiuae/Falcon3-7B-Instruct"
}


class GenerationModel:
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

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def get_chat_prompt(self, prompt):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def get_formatted_prompt(self, prompt):
        chat_prompt = self.get_chat_prompt(prompt)
        return self.tokenizer.apply_chat_template(
            chat_prompt,
            tokenize=False,
            add_generation_prompt=True
        )

    def _make_sampling_params(self, **generation_args):
        do_sample = generation_args.get("do_sample", False)
        max_tokens = generation_args.get("max_new_tokens", 512)
        n = generation_args.get("num_return_sequences", 1)

        if not do_sample:
            # Greedy decoding
            temperature = 0.0
            top_p = 1.0
            top_k = -1
        else:
            temperature = generation_args.get("temperature") or 1.0
            top_p = generation_args.get("top_p") or 1.0
            top_k = generation_args.get("top_k") or -1

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            n=n,
        )

    def get_generated_batch(self, prompts, **generation_args):
        """Generate outputs for a list of prompts in one batched vLLM call."""
        sampling_params = self._make_sampling_params(**generation_args)
        formatted_prompts = [self.get_formatted_prompt(p) for p in prompts]
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        return [[completion.text for completion in req.outputs] for req in outputs]

    def get_generated(self, prompt, **generation_args):
        """Generate outputs for a single prompt."""
        return self.get_generated_batch([prompt], **generation_args)[0]
