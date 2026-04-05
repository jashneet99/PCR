from typing import List, Dict, Optional


def get_prompt_template(prompt: str, dataset: str, stage: str) -> str:
    if prompt == 'zs':
        from prompts.zero_shot import esnli_prompt, comve_prompt, ecqa_prompt
    elif prompt == 'fs':
        from prompts.few_shot import esnli_prompt, comve_prompt, ecqa_prompt
    else:
        raise ValueError(f"Unsupported prompt type: {prompt}")

    mapping = {
        "esnli": esnli_prompt,
        "comve": comve_prompt,
        "ecqa": ecqa_prompt
    }

    if dataset not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset}")

    module = mapping[dataset]

    return getattr(module, f"{stage}_generation_prompt")


def fill_prompt_template(stage: str, prompt: str, item: Dict, top_k: Optional[int] = None) -> Optional[str]:
    prompt = prompt.format(**item)
    
    formatted_options = format_options(item["choices"])
    prompt = prompt.replace("[OPTIONS]", formatted_options)
    
    if stage == 'answer':
        return prompt
    
    elif stage in ['explanation', 'iw_feedback']:
        prompt = prompt.replace("[LABEL]", item["answer"]["final"])
        return prompt
    
    elif stage == 'nl_feedback':
        prompt = prompt.replace("[LABEL]", item["answer"]["final"])
        prompt = prompt.replace("[EXPLANATION]", item["explanation"]["final"])
        return prompt
    
    elif stage == 'nl_refinement':
        prompt = prompt.replace("[LABEL]", item["answer"]["final"])
        prompt = prompt.replace("[EXPLANATION]", item["explanation"]["final"])
        prompt = prompt.replace("[FEEDBACK]", item["nl_feedback"]["final"])
        return prompt
    
    elif stage in ['iw_refinement', 'aiw_ig_refinement', 'aiw_attn_refinement', 'iw_rand_refinement']:
        prompt = prompt.replace("[LABEL]", item["answer"]["final"])
        prompt = prompt.replace("[EXPLANATION]", item["explanation"]["final"])
        
        if stage == 'iw_refinement':
            important_words = item["iw_feedback"]["final"][0]
        elif stage == 'aiw_ig_refinement':
            important_words = item["aiw_ig_feedback"]["merged_sorted"]["words"]
        elif stage == 'aiw_attn_refinement':
            important_words = item["aiw_attn_feedback"]["merged_sorted"]["words"]
        elif stage == 'iw_rand_refinement':
            important_words = item["iw_rand_feedback"]["final"]
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        
        if top_k is not None:
            important_words = important_words[:top_k]
            
        important_words_str = '\n'.join(
            f"{idx + 1}. {word}" for idx, word in enumerate(important_words)
        )
        prompt = prompt.replace("[FEEDBACK]", important_words_str)
        return prompt
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def format_options(options: List[str]) -> str:
    choice_letters = ["A", "B", "C", "D", "E"]
    return "\n".join([f"({ch}) {opt[0].upper() + opt[1:]}" for ch, opt in zip(choice_letters, options)])


def is_valid_answer(answer: str, dataset: str) -> bool:
    valid_choices = {
        "esnli": {"A", "B", "C"},
        "comve": {"A", "B"},
        "ecqa": {"A", "B", "C", "D", "E"},
    }
    return answer in valid_choices.get(dataset, {"A", "B", "C"})
