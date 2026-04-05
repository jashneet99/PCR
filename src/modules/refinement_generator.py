import re
import random
from typing import Dict, List, Tuple
from omegaconf import DictConfig
from modules.generator.generator import GeneralGenerator
from modules.utils import get_prompt_template


def parse(output: str) -> str:
    match = re.search(r"Refined Explanation:\s*(.*)", output, re.DOTALL)
    if match:
        return match.group(1)
    return output.strip()


def apply_voting(candidates: List[str], strategy: str) -> Tuple[str, List[int]]:
    if strategy == "random":
        random_index = random.randint(0, len(candidates) - 1)
        return candidates[random_index], [random_index]
    else:
        raise ValueError(
            f"Unsupported voting strategy for final refinement selection in self-consistency decoding: {strategy}")


class RefinementGenerator:
    def __init__(self, config: DictConfig, model):
        self.feedback_type = config.feedback.type
        self.stage = f"{self.feedback_type}_refinement"
        decoding = config.decoding.type
        if decoding == "sc":
            generation_args = config.generation.refinement.sc.generation_args
            self_consistency = True
            voting_strategy = config.generation.refinement.sc.voting_strategy
        else:
            generation_args = config.generation.refinement.gd.generation_args
            self_consistency = False
            voting_strategy = None
        if self.feedback_type == 'nl':
            refinement_top_k = None
            prompt_template = get_prompt_template(config.prompt.type, config.dataset.name, "nl_refinement")
        else:
            refinement_top_k = config.feedback.top_k
            prompt_template = get_prompt_template(config.prompt.type, config.dataset.name, "iw_refinement")
        self.generator = GeneralGenerator(
            model=model,
            generation_args=generation_args,
            prompt_template=prompt_template,
            parse_fn=parse,
            stage=self.stage,
            answer_validator=None,
            top_k=refinement_top_k,
            self_consistency=self_consistency,
            voting_strategy=voting_strategy,
            voting_fn=apply_voting
        )

    def __call__(self, item: Dict) -> Dict:
        if self.feedback_type == 'nl':
            if item['nl_feedback'] is None or item['nl_feedback']['final'] is None:
                item[self.stage] = None
                return item
            return self.generator(item)
        elif self.feedback_type == 'iw':
            if (item['explanation'] is None or item['explanation']['final'] is None
                    or item['iw_feedback'] is None or item['iw_feedback']['final'] is None):
                item[self.stage] = None
                return item
            return self.generator(item)
        elif self.feedback_type == 'aiw_ig':
            if item['explanation'] is None or item['explanation']['final'] is None or item['aiw_ig_feedback'] is None:
                item[self.stage] = None
                return item
            return self.generator(item)
        elif self.feedback_type == 'aiw_attn':
            if item['explanation'] is None or item['explanation']['final'] is None or item['aiw_attn_feedback'] is None:
                item[self.stage] = None
                return item
            return self.generator(item)
        elif self.feedback_type == 'iw_rand':
            if item['explanation'] is None or item['explanation']['final'] is None or item['iw_rand_feedback'] is None:
                item[self.stage] = None
                return item
            return self.generator(item)
        else:
            raise ValueError(f"Unsupported feedback type: {self.feedback_type}")
