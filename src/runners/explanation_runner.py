import os
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from model.model import GenerationModel
from modules.explanation_generator import ExplanationGenerator
from runners.utils import load_config


def main(base):
    # === Load config ===
    config = load_config()
    
    dataset_type = config.dataset.type
    prompt_type = config.prompt.type
    dataset_name = config.dataset.name
    model_name = config.model.name
    decoding_type = config.decoding.type
    
    # === Load model ===
    model = GenerationModel(model_name)
    
    # === Load generator ===
    generator = ExplanationGenerator(config, model)
    
    # === Construct input and output paths  ===
    base_dir = f"{base}/{dataset_type}/{prompt_type}-{dataset_name}-{model_name}"
    counter_suffix = "" if dataset_type == "original" else "_counter"
    
    input_path = f"{base_dir}/answer_gd{counter_suffix}.json"
    output_path = f"{base_dir}/explanation_{decoding_type}.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # === Load input data ===
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # === Generation ===
    results = generator.batch_call(data)
        
    # === Saving ===
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
        
if __name__ == '__main__':
    base = "experiments"
    main(base)
