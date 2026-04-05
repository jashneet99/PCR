from typing import Dict, List, Optional, Callable, Any

from model.model import GenerationModel
from modules.utils import fill_prompt_template


class GeneralGenerator:
    def __init__(
            self,
            model: GenerationModel,
            generation_args: Dict,
            prompt_template: str,
            parse_fn: Callable[[str], Any],
            stage: str,
            answer_validator: Optional[Callable[[str], bool]] = None,
            top_k: Optional[int] = None,
            self_consistency: bool = False,
            voting_strategy: Optional[str] = None,
            voting_fn: Optional[Callable[[List[Any], str], Any]] = None
    ):
        self.model = model
        self.generation_args = generation_args
        self.prompt_template = prompt_template
        self.parse_fn = parse_fn
        self.stage = stage
        self.answer_validator = answer_validator
        self.top_k = top_k
        self.self_consistency = self_consistency
        self.voting_strategy = voting_strategy
        self.voting_fn = voting_fn
        
        if self.self_consistency:
            if not generation_args.get("do_sample", False):
                raise ValueError("Self-consistency decoding requires 'do_sample=True'")
            if generation_args.get("num_return_sequences", 1) <= 1:
                raise ValueError("Self-consistency decoding requires 'num_return_sequences > 1'")
            if voting_strategy is None:
                raise ValueError("Voting strategy must be specified for self-consistency.")
            if voting_fn is None:
                raise ValueError("Voting function must be specified for self-consistency.")
            
    def __call__(
            self,
            item: Dict
    ):
        prompt = fill_prompt_template(stage=self.stage, prompt=self.prompt_template, item=item, top_k=self.top_k)
        
        outputs = self.model.get_generated(prompt, **self.generation_args)
        
        parsed = []
        valid_indices = []
        selected_indices = []
        
        for i, o in enumerate(outputs):
            p = self.parse_fn(o)
            if p is not None and (not self.answer_validator or self.answer_validator(p)):
                parsed.append(p)
                valid_indices.append(i)
                
        if not parsed:
            final = None
        else:
            if self.self_consistency:
                final, selected_indices = self.voting_fn(parsed, self.voting_strategy)
            else:
                final, selected_indices = parsed[0], []
                
        item[self.stage] = {
            'prompt': prompt,
            'outputs': outputs,
            'parsed': None if not parsed else parsed,
            'valid_indices': None if not valid_indices else valid_indices,
            'selected_indices': None if not selected_indices else selected_indices,
            'final': final
        }
        return item
