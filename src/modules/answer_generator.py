import re
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig
from model.model import GenerationModel
from modules.generator.generator import GeneralGenerator
from modules.utils import get_prompt_template, is_valid_answer


def parse(output: str) -> Optional[str]:
    match = re.search(r"Answer:\s*\(?([A-Z])\)?", output)
    if match:
        return match.group(1)
    return None


def apply_voting(candidates: List[str], strategy: str) -> Tuple[str, List[int]]:
    if strategy == "random":
        random_index = random.randint(0, len(candidates) - 1)
        return candidates[random_index], [random_index]
    elif strategy == "majority":
        counter = Counter(candidates)
        most_common = counter.most_common(1)[0][0]
        indices = [i for i, c in enumerate(candidates) if c == most_common]
        return most_common, indices
    else:
        raise ValueError(
            f"Unsupported voting strategy for final answer selection in self-consistency decoding: {strategy}")


class AnswerGenerator:
    def __init__(self, config: DictConfig, model: GenerationModel):
        self.stage = "answer"
        self.decoding = config.decoding.type
        if self.decoding == "sc":
            generation_args = config.generation.answer.sc.generation_args
            self_consistency = True
            voting_strategy = config.generation.answer.sc.voting_strategy
        else:
            generation_args = config.generation.answer.gd.generation_args
            self_consistency = False
            voting_strategy = None
        prompt_template = get_prompt_template(config.prompt.type, config.dataset.name, self.stage)
        answer_validator = lambda a: is_valid_answer(a, config.dataset.name)
        self.generator = GeneralGenerator(
            model=model,
            generation_args=generation_args,
            prompt_template=prompt_template,
            parse_fn=parse,
            stage=self.stage,
            answer_validator=answer_validator,
            top_k= None,
            self_consistency=self_consistency,
            voting_strategy=voting_strategy,
            voting_fn=apply_voting
        )
        
    def __call__(
            self,
            item: Dict
    ) -> Dict:
        return self.generator(item=item)
