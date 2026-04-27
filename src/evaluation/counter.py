import os
import json
from datetime import datetime
from tabulate import tabulate
from utils import setup_loggers


def counter_stats(org_path, ct_path, save_path):
    with open(org_path, 'r', encoding='utf-8') as f:
        org_data = json.load(f)
    with open(ct_path, 'r', encoding='utf-8') as f:
        ct_data = json.load(f)
        
    total = 0
    valid = 0
    counter = 0
    counter_data = []
    
    for ct_item in ct_data:
        total += 1
        
        idx = ct_item['idx']
        eidx = ct_item['eidx']
        
        org_pred = org_data[idx]["answer"]["final"]
        ct_pred = ct_item["answer"]["final"]
        
        if org_pred is None or ct_pred is None:
            continue
        
        valid += 1
        
        if org_pred != ct_pred:
            counter += 1
            counter_data.append(ct_item)
            
    counter_rate = counter / valid
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(counter_data, f, indent=4, ensure_ascii=False)
        
    return total, valid, counter, counter_rate


if __name__ == '__main__':
    base = 'experiments'
    
    prompt_types = ['zs']
    datasets = ['comve', 'ecqa']
    models = ['qwen']
    answer_sources = ['gd']
    
    log_path = f"logs/counter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, blank_logger = setup_loggers(log_path)
    
    for prompt_type in prompt_types:
        for dataset_name in datasets:
            table = []
            for model_name in models:
                for answer_source in answer_sources:
                    org_path = f"{base}/original/{prompt_type}-{dataset_name}-{model_name}/answer_{answer_source}.json"
                    ct_path = f"{base}/counterfactual/{prompt_type}-{dataset_name}-{model_name}/answer_{answer_source}.json"
                    save_path = f"{base}/counterfactual/{prompt_type}-{dataset_name}-{model_name}/answer_{answer_source}_counter.json"
                    
                    total, valid, counter, counter_rate = counter_stats(org_path, ct_path, save_path)
                    
                    table.append([
                        prompt_type,
                        dataset_name,
                        model_name,
                        answer_source,
                        total,
                        valid,
                        counter,
                        f"{counter_rate * 100:.2f}%"
                    ])
            headers = ['Prompt', 'Dataset', 'Model', 'Answer', "Total", "Valid", "Counter", "Counter Rate"]
            blank_logger.info(tabulate(table, headers=headers, tablefmt="pretty"))
            blank_logger.info("")
        blank_logger.info("")
