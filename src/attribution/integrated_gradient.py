import torch
from collections import defaultdict
from typing import Dict, List, Tuple
from captum.attr import IntegratedGradients

from model.model import GenerationModel

from attribution.utils import (
    find_target_token_span,
    find_field_token_span,
    aggregate_attributions_target,
    aggregate_attributions_word
)

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


class IntegratedGradientsAttribution:
    def __init__(
            self,
            model: GenerationModel,
            dataset: str,
            model_name: str,
            target_agg_method: str = "abs_mean",
            word_agg_method: str = "sum"
    ):
        self.model = model
        self.dataset = dataset
        self.target_agg_method = target_agg_method
        self.word_agg_method = word_agg_method
        self.model_name = model_name
        
        if self.model_name == "falcon":
            self.n_steps = 500
        elif self.model_name == "llama":
            self.n_steps = 500
        elif self.model_name == "mistral":
            self.n_steps = 500
        elif self.model_name == "qwen":
            self.n_steps = 1000
            
        self.field_map = {
            "esnli": ["premise", "hypothesis"],
            "comve": ["sentence0", "sentence1"],
            "ecqa": ["question"]
        }

    def __call__(
            self,
            item: Dict
    ) -> Dict:
        item['aiw_ig_feedback'] = {}
        
        input_text = self.model.get_formatted_prompt(item["answer"]["prompt"])
        generated_text = item["answer"]["outputs"][0]
        target_text = item["answer"]["final"]
        input_len = self.model.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)["input_ids"].size(1)
        
        target_attr_res, delta_res = self._compute_ig(input_text, generated_text, target_text, self.n_steps)
        item['aiw_ig_feedback']['delta_res'] = delta_res
        
        formatted_target_attr_res = [attr_res[:input_len] for attr_res in target_attr_res]
        aggregated_target_attr_res = aggregate_attributions_target(formatted_target_attr_res, self.target_agg_method)
        
        for field in self.field_map[self.dataset]:
            filed_start, field_end = find_field_token_span(input_text, item[field], self.model)
            field_tokens = aggregated_target_attr_res[filed_start: field_end + 1]
            item['aiw_ig_feedback'][field] = aggregate_attributions_word(field_tokens, self.word_agg_method)
            
        all_word_score_pairs = []
        merged_word_score_pairs = defaultdict(float)
        
        for field in self.field_map[self.dataset]:
            word_score_pair = item['aiw_ig_feedback'][field]
            for word, score in zip(word_score_pair['words'], word_score_pair['scores']):
                word = word.lower()
                all_word_score_pairs.append((word, score))
                merged_word_score_pairs[word] += score
                
        all_sorted_word_score_pairs = sorted(all_word_score_pairs, key=lambda x: x[1], reverse=True)
        item['aiw_ig_feedback']['all_sorted'] = {
            "words": [word for word, _ in all_sorted_word_score_pairs],
            "scores": [score for _, score in all_sorted_word_score_pairs]
        }
        
        merged_sorted_word_score_pairs = sorted(merged_word_score_pairs.items(), key=lambda x: x[1], reverse=True)
        item['aiw_ig_feedback']['merged_sorted'] = {
            "words": [word for word, _ in merged_sorted_word_score_pairs],
            "scores": [score for _, score in merged_sorted_word_score_pairs]
        }
        
        all_filtered_word_score_pairs = []
        for word, score in all_sorted_word_score_pairs:
            if word not in STOPWORDS:
                all_filtered_word_score_pairs.append((word, score))
                
        merged_filtered_word_score_pairs = []
        for word, score in merged_sorted_word_score_pairs:
            if word not in STOPWORDS:
                merged_filtered_word_score_pairs.append((word, score))
                
        item['aiw_ig_feedback']['all_filtered'] = {
            "words": [word for word, _ in all_filtered_word_score_pairs],
            "scores": [score for _, score in all_filtered_word_score_pairs]
        }
        
        item['aiw_ig_feedback']['merged_filtered'] = {
            "words": [word for word, _ in merged_filtered_word_score_pairs],
            "scores": [score for _, score in merged_filtered_word_score_pairs]
        }
        
        return item

    def _compute_ig(
            self,
            input_text: str,
            generated_text: str,
            target_text: str,
            n_steps: int
    ) -> Tuple[List[List[Tuple[int, str, float]]], List[float]]:
        
        full_text = input_text + generated_text
        inputs = self.model.tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        target_start, target_end = find_target_token_span(
            input_text,
            generated_text,
            target_text,
            self.model
        )
        
        def forward_func(input_embeds, attention_mask, target_token_id):
            outputs = self.model.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            return next_token_logits[:, target_token_id]
        
        def get_baseline(input_ids, model=self.model.model, tokenizer=self.model.tokenizer, method="eos"):
            if method == "zero":
                baseline_embeds = torch.zeros_like(model.get_input_embeddings()(input_ids))
            elif method == "pad":
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                baseline_input = torch.full_like(input_ids, pad_token_id)
                baseline_embeds = model.get_input_embeddings()(baseline_input)
            elif method == "eos":
                eos_token_id = tokenizer.eos_token_id
                baseline_input = torch.full_like(input_ids, eos_token_id)
                baseline_embeds = model.get_input_embeddings()(baseline_input)
            elif method == "mean":
                vocab_embeds = model.get_input_embeddings().weight
                mean_embed = vocab_embeds.mean(dim=0, keepdim=True)  # (1, embed_dim)
                baseline_embeds = mean_embed.expand_as(model.get_input_embeddings()(input_ids))
            else:
                raise ValueError(f"Unknown baseline method: {method}")
            return baseline_embeds
        
        def safe_attribute(ig, input_embeds, baseline, attention_mask, target_token_id, n_steps, initial_batch_size):
            batch_size = initial_batch_size
            while batch_size >= 1:
                try:
                    attributions, delta = ig.attribute(
                        inputs=input_embeds,
                        baselines=baseline,
                        additional_forward_args=(attention_mask, target_token_id),
                        n_steps=n_steps,
                        return_convergence_delta=True,
                        internal_batch_size=batch_size
                    )
                    return attributions, delta
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        batch_size //= 2
                        torch.cuda.empty_cache()
                    else:
                        raise e
            raise RuntimeError("Failed to attribute even with internal_batch_size=1.")
        
        ig = IntegratedGradients(forward_func)
        
        target_attribution_results = []
        delta_results = []
        for target_token_pos in range(target_start, target_end + 1):
            target_token_id = input_ids[0][target_token_pos].item()
            # target_token_str = self.model.tokenizer.decode(target_token_id)
            # print(f"Analyzing Token: '{target_token_str}', Id: '{target_token_id}', Position: {target_token_pos}")
            
            causual_input_ids = input_ids[:, :target_token_pos]
            causual_attention_mask = attention_mask[:, :target_token_pos]
            
            causual_input_embeds = self.model.model.get_input_embeddings()(causual_input_ids).detach().requires_grad_()
            
            baseline = get_baseline(causual_input_ids, method="eos")
            
            attributions, delta = safe_attribute(
                ig,
                causual_input_embeds,
                baseline,
                causual_attention_mask,
                target_token_id,
                n_steps=n_steps,
                initial_batch_size=50
            )
            
            token_attributions = attributions.sum(dim=-1).squeeze(0)
            scores = token_attributions.detach().cpu().numpy()
            
            tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0)[:target_token_pos])
            token_ids = self.model.tokenizer.convert_tokens_to_ids(tokens)
            
            result = []
            for token_id, token, score in zip(token_ids, tokens, scores):
                result.append((token_id, token, score))
                
            target_attribution_results.append(result)
            delta_results.append(delta.item())
            
        return target_attribution_results, delta_results
