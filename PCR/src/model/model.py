import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LLM_MODELS = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "falcon":  "tiiuae/Falcon3-7B-Instruct"
}


class GenerationModel:
    """
    Extended version of SR-NLE's GenerationModel.
    Adds get_margin() for PCR's probabilistic faithfulness check.

    New method:
        get_margin(explanation, y, y_prime)
            → feeds explanation ALONE to the model (no original input x)
            → extracts logits for y and y_prime answer tokens
            → applies softmax → returns Δt = P(y|e) - P(y'|e)
    """

    def __init__(self, model_name):
        self.model_id = LLM_MODELS[model_name]

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.device = next(self.model.parameters()).device
        self.system_prompt = "You are a helpful assistant!"

    # ------------------------------------------------------------------ #
    #  Original SR-NLE methods (unchanged)                                #
    # ------------------------------------------------------------------ #

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def get_chat_prompt(self, prompt):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]
        return messages

    def get_formatted_prompt(self, prompt):
        chat_prompt = self.get_chat_prompt(prompt)
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt

    def get_inputs(self, prompt):
        formatted_prompt = self.get_formatted_prompt(prompt)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        return inputs

    def get_generated(self, prompt, **generation_args):
        inputs = self.get_inputs(prompt)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_args
            )
        decoded_outputs = [
            self.tokenizer.decode(
                output[inputs['input_ids'].size(1):],
                skip_special_tokens=True
            )
            for output in outputs
        ]
        return decoded_outputs

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
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_args
            )
        decoded = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].size(1):],
            skip_special_tokens=True
        )
        return decoded.strip()

    def set_eval_mode(self):
        self.model.eval()

    def set_train_mode(self):
        self.model.train()

    # ------------------------------------------------------------------ #
    #  PCR Extension — get_margin()                                       #
    # ------------------------------------------------------------------ #

    def get_answer_token_id(self, answer_label: str) -> int:
        """
        Returns the first token ID of the answer label string.
        e.g. "A" → token_id for "A"
             "Sentence 0" → token_id for "Sentence"
        Used to extract logits at the answer position.
        """
        tokens = self.tokenizer.encode(
            " " + answer_label,   # space prefix handles BPE tokenisation
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
            Δt =  0.0  → explanation is ambiguous (confused between y and y')
            Δt = -1.0  → explanation pushes towards wrong prediction

        Args:
            explanation : the explanation text (DE or CE) — no original input
            y           : the factual answer label  (e.g. "A", "Sentence 1")
            y_prime     : the alternative answer label

        Returns:
            Δt (float) in range [-1, +1]
        """
        # Step 1 — Build prompt with ONLY the explanation (original x is hidden)
        prompt = (
            f"Based only on the following explanation, predict the answer.\n\n"
            f"Explanation: {explanation}\n\n"
            f"Answer (choose one):"
        )

        # Step 2 — Tokenise and get model logits
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)
            # logits shape: [1, seq_len, vocab_size]
            # We want the logits at the LAST position (where answer token goes)
            last_logits = output.logits[0, -1, :]  # [vocab_size]

        # Step 3 — Extract logits for y and y' answer tokens
        y_token_id       = self.get_answer_token_id(y)
        y_prime_token_id = self.get_answer_token_id(y_prime)

        y_logit       = last_logits[y_token_id].item()
        y_prime_logit = last_logits[y_prime_token_id].item()

        # Step 4 — Apply softmax over just these two logits
        # (binary softmax = sigmoid equivalent for 2-class case)
        logits_pair = torch.tensor([y_logit, y_prime_logit])
        probs = F.softmax(logits_pair, dim=0)

        p_y       = probs[0].item()   # P(y | explanation)
        p_y_prime = probs[1].item()   # P(y' | explanation)

        # Step 5 — Compute margin Δt
        delta_t = p_y - p_y_prime

        return delta_t
