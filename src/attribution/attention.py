import torch
from collections import defaultdict
from typing import Dict, List, Tuple

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


class AttentionAttribution:
    def __init__(
            self,
            model: GenerationModel,
            dataset: str,
            aggregation: str = "last",
            target_agg_method: str = "abs_mean",
            word_agg_method: str = "sum"
    ):
        self.model = model
        self.dataset = dataset
        self.aggregation = aggregation
        self.target_agg_method = target_agg_method
        self.word_agg_method = word_agg_method
        
        self.field_map = {
            "esnli": ["premise", "hypothesis"],
            "comve": ["sentence0", "sentence1"],
            "ecqa": ["question"]
        }
        
    def __call__(
            self,
            item: Dict
    ) -> Dict:
        item['aiw_attn_feedback'] = {}
        
        input_text = self.model.get_formatted_prompt(item["answer"]["prompt"])
        generated_text = item["answer"]["outputs"][0]
        target_text = item["answer"]["final"]
        input_len = self.model.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)["input_ids"].size(1)
        
        target_attr_res = self._compute_attention(input_text, generated_text, target_text)
        formatted_target_attr_res = [attr_res[:input_len] for attr_res in target_attr_res]
        aggregated_target_attr_res = aggregate_attributions_target(formatted_target_attr_res, self.target_agg_method)
        
        for field in self.field_map[self.dataset]:
            field_start, field_end = find_field_token_span(input_text, item[field], self.model)
            field_tokens = aggregated_target_attr_res[field_start: field_end + 1]
            item['aiw_attn_feedback'][field] = aggregate_attributions_word(field_tokens, self.word_agg_method)
            
        all_word_score_pairs = []
        merged_word_score_pairs = defaultdict(float)
        for field in self.field_map[self.dataset]:
            word_score_pair = item['aiw_attn_feedback'][field]
            for word, score in zip(word_score_pair['words'], word_score_pair['scores']):
                word = word.lower()
                all_word_score_pairs.append((word, score))
                merged_word_score_pairs[word] += score
                
        all_sorted_word_score_pairs = sorted(all_word_score_pairs, key=lambda x: x[1], reverse=True)
        item['aiw_attn_feedback']['all_sorted'] = {
            "words": [word for word, _ in all_sorted_word_score_pairs],
            "scores": [score for _, score in all_sorted_word_score_pairs]
        }
        
        merged_sorted_word_score_pairs = sorted(merged_word_score_pairs.items(), key=lambda x: x[1], reverse=True)
        item['aiw_attn_feedback']['merged_sorted'] = {
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
                
        item['aiw_attn_feedback']['all_filtered'] = {
            "words": [word for word, _ in all_filtered_word_score_pairs],
            "scores": [score for _, score in all_filtered_word_score_pairs]
        }
        
        item['aiw_attn_feedback']['merged_filtered'] = {
            "words": [word for word, _ in merged_filtered_word_score_pairs],
            "scores": [score for _, score in merged_filtered_word_score_pairs]
        }
        
        return item
    
    def _compute_attention(
            self,
            input_text: str,
            generated_text: str,
            target_text: str
    ) -> List[List[Tuple[int, str, float]]]:
        
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
        
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False
            )
            all_attentions = [attn.to(torch.float32) for attn in outputs.attentions]  # List of [B, H, S, S]
            
        if self.aggregation == "last":
            layer_attn = all_attentions[-1]  # shape: [B, H, S, S]
            avg_attn = layer_attn[0].mean(dim=0)  # avg over heads → shape: [S, S]
        elif self.aggregation == "avg":
            # stack all layers: [num_layers, batch_size, num_heads, seq_len, seq_len]
            stacked = torch.stack(all_attentions, dim=0)  # shape: [L, B, H, S, S]
            avg_attn = stacked.mean(dim=(0, 1, 2))  # avg over L, B, H → shape: [S, S]
        else:
            raise ValueError("Unknown aggregation method: {}".format(self.aggregation))
        
        assert avg_attn.shape[0] == avg_attn.shape[1]
        target_attribution_results = []
        
        for target_token_pos in range(target_start, target_end + 1):
            
            scores = avg_attn[target_token_pos, :].detach().cpu().numpy()
            
            tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            token_ids = self.model.tokenizer.convert_tokens_to_ids(tokens)
            
            result = []
            for token_id, token, score in zip(token_ids, tokens, scores):
                result.append((token_id, token, score))
                
            target_attribution_results.append(result)
            
        return target_attribution_results
