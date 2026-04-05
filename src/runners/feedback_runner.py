import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from model.model import GenerationModel
from modules.feedback_generator import FeedbackGenerator
from runners.utils import load_config


def main(base):
    # === Load config ===
    config = load_config()
    
    dataset_type = config.dataset.type
    prompt_type = config.prompt.type
    dataset_name = config.dataset.name
    model_name = config.model.name
    feedback_type = config.feedback.type
    seed = config.seed
    iteration = config.iteration
    
    # === Construct input and output paths  ===
    base_dir = f"{base}/{dataset_type}/{prompt_type}-{dataset_name}-{model_name}"
    seed_suffix = f"_{seed}" if feedback_type == "iw_rand" else ""
    
    if iteration == 0:
        input_path = f"{base_dir}/explanation_gd.json"
    else:
        input_path = f"{base_dir}/iter{int(iteration) - 1}_refinement_{feedback_type}{seed_suffix}.json"
        
    output_path = f"{base_dir}/iter{iteration}_feedback_{feedback_type}{seed_suffix}.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # === Load input data ===
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # === Generation ===
    results = []
    if iteration == 0:
        model = GenerationModel(model_name)
        generator = FeedbackGenerator(config, model)

        for item in tqdm(data, desc="Generating feedback"):
            item = generator(item)
            results.append(item)
    else:
        if feedback_type in ['iw', 'aiw_ig', 'aiw_attn', 'iw_rand']:
            for item in tqdm(data, desc="Generating feedback"):
                item["explanation"] = item[f"{feedback_type}_refinement"]
                del item[f"{feedback_type}_refinement"]
                results.append(item)
        else:
            model = GenerationModel(model_name)
            generator = FeedbackGenerator(config, model)
            for item in tqdm(data, desc="Generating feedback"):
                item["explanation"] = item[f"{feedback_type}_refinement"]
                del item[f"{feedback_type}_refinement"]
                item = generator(item)
                results.append(item)
                
    # === Saving ===
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
        
if __name__ == '__main__':
    base = "experiments"
    main(base)
