import re
import torch
import random
from omegaconf import DictConfig
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from model.model import GenerationModel
from modules.generator.generator import GeneralGenerator
from modules.utils import get_prompt_template
from attribution.attention import AttentionAttribution
from attribution.integrated_gradient import IntegratedGradientsAttribution
from attribution.random import RandomIWF


def parse_nl(output: str) -> Optional[str]:
    match = re.search(r"Feedback:\s*(.*)", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_iw(
        output: str,
        sort: bool = True,
        remove_duplicates: bool = True
) -> Tuple[List[str], List[float]]:
    words = []
    scores = []

    lines = output.strip().splitlines()
    pattern = re.compile(r"^\d+\.\s*(\S+),\s*(\d+)(?:\s*\(.*?\))?\s*$")

    if remove_duplicates:
        score_map = defaultdict(float)
        for line in lines:
            match = pattern.match(line.strip())
            if match:
                word = match.group(1).strip().lower()
                score = float(match.group(2).strip())
                score_map[word] = max(score_map[word], score)
        words_scores = list(score_map.items())
    else:
        words_scores = []
        for line in lines:
            match = pattern.match(line.strip())
            if match:
                word = match.group(1).strip().lower()
                score = float(match.group(2).strip())
                words_scores.append((word, score))
                
    if sort:
        words_scores.sort(key=lambda x: -x[1])
        
    if words_scores:
        words, scores = zip(*words_scores)
    else:
        words, scores = [], []
        
    return list(words), list(scores)


def apply_voting_nl(candidates: List[str], strategy: str) -> Tuple[str, List[int]]:
    if strategy == "random":
        random_index = random.randint(0, len(candidates) - 1)
        return candidates[random_index], [random_index]
    else:
        raise ValueError(
            f"Unsupported voting strategy for final natural language feedback selection in self-consistency decoding: {strategy}")


def apply_voting_iw(candidates: List[List[str]], strategy: str) -> Tuple[List[str], List[int]]:
    if strategy == "random":
        random_index = random.randint(0, len(candidates) - 1)
        return candidates[random_index], [random_index]
    else:
        raise ValueError(
            f"Unsupported voting strategy for final important words selection in self-consistency decoding: {strategy}")


class FeedbackGenerator:
    def __init__(self, config: DictConfig, model: GenerationModel):
        self.feedback_type = config.feedback.type
        self.stage = f"{self.feedback_type}_feedback"
        if self.feedback_type in ['nl', 'iw']:
            decoding = config.decoding.type
            if decoding == "sc":
                generation_args = config.generation.feedback.sc.generation_args
                self_consistency = True
                voting_strategy = config.generation.feedback.sc.voting_strategy
            else:
                generation_args = config.generation.feedback.gd.generation_args
                self_consistency = False
                voting_strategy = None
            prompt_template = get_prompt_template(config.prompt.type, config.dataset.name, self.stage)
            self.generator = GeneralGenerator(
                model=model,
                generation_args=generation_args,
                prompt_template=prompt_template,
                parse_fn=parse_nl if self.feedback_type == "nl" else parse_iw,
                stage=self.stage,
                answer_validator=None,
                top_k=None,
                self_consistency=self_consistency,
                voting_strategy=voting_strategy,
                voting_fn=apply_voting_nl if self.feedback_type == "nl" else apply_voting_iw
            )
        elif self.feedback_type == "aiw_ig":
            self.generator = IntegratedGradientsAttribution(
                model=model,
                dataset=config.dataset.name,
                model_name=config.model.name
            )
        elif self.feedback_type == "aiw_attn":
            self.generator = AttentionAttribution(
                model=model,
                dataset=config.dataset.name
            )
        elif self.feedback_type == "iw_rand":
            self.generator = RandomIWF(
                dataset = config.dataset.name,
                seed= config.seed
            )

    def __call__(
            self,
            item: Dict
    ) -> Dict:
        torch.cuda.empty_cache()
        if item['explanation'] is None or item['explanation']['final'] is None:
            item[self.stage] = None
            return item
        return self.generator(item)

    def batch_call(self, items: List[Dict]) -> List[Dict]:
        # Attribution-based feedback types (aiw_ig, aiw_attn) require HF model
        # gradients and are not compatible with vLLM — fall back to per-item.
        if self.feedback_type in ['aiw_ig', 'aiw_attn', 'iw_rand']:
            for item in items:
                self(item)
            return items

        valid_items = []
        for item in items:
            if item['explanation'] is None or item['explanation']['final'] is None:
                item[self.stage] = None
            else:
                valid_items.append(item)
        if valid_items:
            self.generator.batch_call(valid_items)
        return items
