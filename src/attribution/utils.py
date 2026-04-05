import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from model.model import GenerationModel


def find_target_token_span(
        input_text: str,
        generated_text: str,
        target_text: str,
        model: GenerationModel
) -> Tuple[int, int]:
    input_ids = model.tokenizer.encode(input_text, add_special_tokens=False)
    generated_ids = model.tokenizer.encode(generated_text, add_special_tokens=False)
    target_ids_no_space = model.tokenizer.encode(target_text, add_special_tokens=False)
    target_ids_with_space = model.tokenizer.encode(" " + target_text, add_special_tokens=False)
    
    start = None
    end = None
    
    for candidates_ids in [target_ids_with_space, target_ids_no_space]:
        for i in range(len(generated_ids) - len(candidates_ids) + 1):
            if generated_ids[i: i + len(candidates_ids)] == candidates_ids:
                start = len(input_ids) + i
                end = start + len(candidates_ids) - 1
                return start, end
            
    if start is None or end is None:
        return -1, -1


def find_field_token_span(
        input_text: str,
        field_text: str,
        model: GenerationModel
) -> Tuple[int, int]:
    input_ids = model.tokenizer.encode(input_text, add_special_tokens=False)
    field_ids = model.tokenizer.encode(" " + field_text + "\n", add_special_tokens=False)
    
    start = None
    end = None
    
    for i in range(len(input_ids) - len(field_ids) + 1):
        if input_ids[i: i + len(field_ids)] == field_ids:
            start = i
            end = start + len(field_ids) - 1
            
    if start is None or end is None:
        return -1, -1
    
    return start, end


def aggregate_attributions_target(
        target_attr_res: List[List[Tuple[int, str, float]]],
        method: str = "abs_mean"
) -> List[Tuple[int, str, float]]:
    
    score_matrix = np.array([
        [score for _, _, score in result] for result in target_attr_res
    ])
    
    aggregated = None
    
    if method == "abs_mean":
        aggregated = np.abs(score_matrix).mean(axis=0)
    elif method == "abs_sum":
        aggregated = np.abs(score_matrix).sum(axis=0)
    elif method == "signed_mean":
        aggregated = score_matrix.mean(axis=0)
    elif method == "signed_sum":
        aggregated = score_matrix.sum(axis=0)
        
    if aggregated is None:
        return []
    
    token_ids = [token_id for token_id, _, _ in target_attr_res[0]]
    tokens = [token for _, token, _ in target_attr_res[0]]
    
    return list(zip(token_ids, tokens, aggregated.tolist()))


def aggregate_attributions_word(
        aggregated_token_res: List[Tuple[int, str, float]],
        method: str = "mean",
) -> Dict[str, List]:

    punctuations = {",", ".", "?", "!"}
    
    words = []
    scores = []
    
    current_word = ""
    current_score = 0.0
    count = 0
    
    for token_id, token, score in aggregated_token_res:
        token = token.strip()
        
        if not token or 'Ċ' in token or '<0x0A>' in token:
            continue
        
        if token in punctuations:
            continue
        
        if token.startswith("Ġ") or token.startswith("▁"):
            if current_word:
                final_score = current_score / count if method == "mean" else current_score
                words.append(current_word)
                scores.append(final_score)
                
            clean_token = token.lstrip("Ġ▁")
            if clean_token and clean_token not in punctuations:
                current_word = clean_token
                current_score = score
                count = 1
            else:
                current_word = ""
                current_score = 0.0
                count = 0
        else:
            current_word += token
            current_score += score
            count += 1
            
    if current_word:
        final_score = current_score / count if method == "mean" else current_score
        words.append(current_word)
        scores.append(final_score)
        
    return {
        "words": words,
        "scores": scores
    }


def plot_word_attributions(
        word_attributions: List[Tuple[str, float]],
        title: str,
        save_path: Optional[str] = None,
        show: bool = False
):
    words, scores = zip(*word_attributions)
    bar_colors = ['green' for score in scores]
    
    x = np.arange(len(words))
    
    plt.figure(figsize=(15, 5))
    plt.bar(x, scores, color=bar_colors)
    
    plt.xticks(ticks=x, labels=words, rotation=45, ha='right')
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()
        
    plt.close()


def plot_attention_matrix(avg_attn, tokens=None):
    attn = avg_attn.detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    if tokens:
        plt.xticks(ticks=range(len(tokens)), labels=tokens, rotation=90)
        plt.yticks(ticks=range(len(tokens)), labels=tokens)
    plt.xlabel("Attended Token")
    plt.ylabel("Query Token")
    plt.title("Average Attention Matrix")
    plt.tight_layout()
    plt.show()
    plt.savefig("attention-matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
