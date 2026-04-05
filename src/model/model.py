import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LLM_MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "falcon": "tiiuae/Falcon3-7B-Instruct"
}


class GenerationModel:
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

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def get_chat_prompt(self, prompt):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
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

    def set_eval_mode(self):
        self.model.eval()

    def set_train_mode(self):
        self.model.train()
