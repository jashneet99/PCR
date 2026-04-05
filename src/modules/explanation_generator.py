import re
import random
import numpy as np
from omegaconf import DictConfig
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from model.model import GenerationModel
from modules.generator.generator import GeneralGenerator
from modules.utils import get_prompt_template

sbert_model = SentenceTransformer("all-mpnet-base-v2")


def parse(output: str) -> Optional[str]:
    match = re.search(r"Explanation:\s*(.*)", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def apply_voting(candidates: List[str], strategy: str) -> Tuple[str, List[int]]:
    if strategy == "random":
        random_index = random.randint(0, len(candidates) - 1)
        return candidates[random_index], [random_index]
    elif strategy == "center":
        if len(candidates) == 1:
            return candidates[0], [0]
        embeddings = sbert_model.encode(candidates)
        center = np.mean(embeddings, axis=0, keepdims=True)
        sims = cosine_similarity(center, embeddings)[0]
        best_index = int(np.argmax(sims))
        return candidates[best_index], [best_index]
    else:
        raise ValueError(
            f"Unsupported voting strategy for final explanation selection in self-consistency decoding: {strategy}")


class ExplanationGenerator:
    def __init__(self, config: DictConfig, model: GenerationModel):
        self.stage = "explanation"
        decoding = config.decoding.type
        if decoding == "sc":
            generation_args = config.generation.explanation.sc.generation_args
            self_consistency = True
            voting_strategy = config.generation.explanation.sc.voting_strategy
        else:
            generation_args = config.generation.explanation.gd.generation_args
            self_consistency = False
            voting_strategy = None
        prompt_template = get_prompt_template(config.prompt.type, config.dataset.name, self.stage)
        self.generator = GeneralGenerator(
            model=model,
            generation_args=generation_args,
            prompt_template=prompt_template,
            parse_fn=parse,
            stage=self.stage,
            answer_validator=None,
            top_k=None,
            self_consistency=self_consistency,
            voting_strategy=voting_strategy,
            voting_fn=apply_voting
        )
        
    def __call__(
            self,
            item: Dict
    ) -> Dict:
        if item["answer"]["final"] is None:
            item[self.stage] = None
            return item
        return self.generator(item=item)
