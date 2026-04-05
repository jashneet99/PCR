import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from model.model import GenerationModel
from modules.answer_generator import AnswerGenerator
from runners.utils import load_config


def main(base):
    # === Load config ===
    config = load_config()
    
    dataset_type = config.dataset.type
    prompt_type = config.prompt.type
    dataset_name = config.dataset.name
    model_name = config.model.name
    decoding_type = config.decoding.type
    num_samples = config.dataset.num_samples
    
    # === Load model ===
    model = GenerationModel(model_name)
    
    # === Load generator ===
    generator = AnswerGenerator(config, model)
    
    # === Construct input and output paths ===
    base_dir = f"{base}/{dataset_type}/{prompt_type}-{dataset_name}-{model_name}"
    
    if dataset_type == 'original':
        input_path = f"data/formatted/{dataset_name}/test.json"
    else:
        input_path = f"data/{dataset_type}/{dataset_name}/ext_org.json"  # Fix: was ext_final.json
        
    output_path = f"{base_dir}/answer_{decoding_type}.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # === Load input data ===
    if dataset_type == 'original':
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)[:num_samples]
    elif dataset_type == 'counterfactual':
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)['extract_edits_dataset'][:num_samples * 10]  # Fix: was * 20
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # === Generation ===
    results = []
    for item in tqdm(data, desc="Generating answer"):
        item = generator(item)
        results.append(item)
        
    # === Saving ===
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
        
if __name__ == '__main__':
    base = "experiments"
    main(base)