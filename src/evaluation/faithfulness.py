import json
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from utils import setup_loggers


def is_word_in_expl(word, expl):
    return word.lower() in expl.lower()


def init_stats(ct_path):
    with open(ct_path, 'r', encoding='utf-8') as f:
        ct_data = json.load(f)
        
    total = 0
    valid = 0
    faith = 0
    unfaith = 0
    
    faith_items = []
    unfaith_items = []
    
    lengths = []
    
    for ct_item in ct_data:
        total += 1
        
        if ct_item["explanation"] is None or ct_item["explanation"]["final"] is None:
            continue
        
        valid += 1
        
        edit_word = ct_item["edit_word"]
        gen_expl = ct_item["explanation"]["final"]
        
        lengths.append(len(gen_expl.split()))
        
        if is_word_in_expl(edit_word, gen_expl):
            faith += 1
            faith_items.append(ct_item)
        else:
            unfaith += 1
            unfaith_items.append(ct_item)
            
    faith_rate = faith / valid
    unfaith_rate = unfaith / valid
    
    res = {
        "total": total,
        "valid": valid,
        "faith": faith,
        "unfaith": unfaith,
        "faith_rate": faith_rate,
        "unfaith_rate": unfaith_rate,
        "lengths": sum(lengths)/ len(lengths),
    }
    return res


def refined_stats(ct_path, feedback_type):
    with open(ct_path, 'r', encoding='utf-8') as f:
        ct_data = json.load(f)
        
    total = 0
    valid = 0
    faith = 0
    unfaith = 0
    
    faith_items = []
    unfaith_items = []
    
    lengths = []
    
    for ct_item in ct_data:
        total += 1
        
        if feedback_type == 'iw' and not ct_item[f"{feedback_type}_feedback"]['final'][0]:
            continue
        
        if ct_item[f"{feedback_type}_refinement"] is None or ct_item[f"{feedback_type}_refinement"]['final'] is None:
            continue
        
        valid += 1
        
        edit_word = ct_item["edit_word"]
        gen_expl = ct_item[f"{feedback_type}_refinement"]['final']
        
        lengths.append(len(gen_expl.split()))
        
        if is_word_in_expl(edit_word, gen_expl):
            faith += 1
            faith_items.append(ct_item)
        else:
            unfaith += 1
            unfaith_items.append(ct_item)
            
    faith_rate = faith / valid
    unfaith_rate = unfaith / valid
    
    res = {
        "total": total,
        "valid": valid,
        "faith": faith,
        "unfaith": unfaith,
        "faith_rate": faith_rate,
        "unfaith_rate": unfaith_rate,
        "lengths": sum(lengths)/ len(lengths),
    }
    return res


if __name__ == '__main__':
    base = "experiments"
    
    dataset_type = "counterfactual"
    prompt_type = "zs"
    
    datasets = ['comve']
    models = ['qwen']
    feedback_types = ['nl']
    explanation_sources = ['gd']
    iterations = [0, 1, 2]
    
    log_path = f"logs/faithfulness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, blank_logger = setup_loggers(log_path)
    
    results = []
    for dataset_name in datasets:
        for model_name in models:
            base_dir = f"{base}/{dataset_type}/{prompt_type}-{dataset_name}-{model_name}"
            # === init explanation ===
            for explanation_source in explanation_sources:
                if explanation_source == 'sc':
                    path = f"{base_dir}/explanation_{explanation_source}_center.json"
                else:
                    path = f"{base_dir}/explanation_{explanation_source}.json"
                    
                res = init_stats(path)
                
                results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Stage": "init",
                    "Iter": "-",
                    "Type": explanation_source,
                    "Total": res["total"],
                    "Valid": res["valid"],
                    "Faith": res["faith"],
                    "Unfaith": res["unfaith"],
                    "Faith Rate": round(res['faith_rate'], 4),
                    "Unfaith Rate": round(res['unfaith_rate'], 4),
                    "Lengths": round(res['lengths'], 4),
                })
                
            # === Refined explanation ===
            for i in iterations:
                for feedback_type in feedback_types:
                    path = f"{base_dir}/iter{i}_refinement_{feedback_type}.json"
                    import os
                    if not os.path.exists(path):
                        continue
                    res = refined_stats(path, feedback_type)
                    
                    results.append({
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "Stage": "refined",
                        "Iter": i,
                        "Type": feedback_type,
                        "Total": res["total"],
                        "Valid": res["valid"],
                        "Faith": res["faith"],
                        "Unfaith": res["unfaith"],
                        "Faith Rate": round(res['faith_rate'], 4),
                        "Unfaith Rate": round(res['unfaith_rate'], 4),
                        "Lengths": round(res['lengths'], 4),
                    })

    df = pd.DataFrame(results)
    csv_path = f"logs/df/faithfulness_results.csv"
    df.to_csv(csv_path, index=False)
    
    for dataset_name, dataset_df in df.groupby("Dataset"):
        blank_logger.info(f"==== {dataset_name} ====\n")
        
        for stage_name, stage_df in dataset_df.groupby("Stage"):
            if stage_name == "refined":
                for iter_num, iter_df in stage_df.groupby("Iter"):
                    for type_name, type_df in iter_df.groupby("Type"):
                        blank_logger.info(f"Stage: {stage_name}, Iter: {iter_num}, Type: {type_name}")
                        blank_logger.info(tabulate(type_df, headers="keys", tablefmt="pretty", showindex=False))
                        blank_logger.info("\n")
            else:
                for type_name, type_df in stage_df.groupby("Type"):
                    blank_logger.info(f"Stage: {stage_name}, Type: {type_name}")
                    blank_logger.info(tabulate(type_df, headers="keys", tablefmt="pretty", showindex=False))
                    blank_logger.info("\n")
                    
    grouped = df.groupby(["Stage", "Iter", "Type"])
    grouped_mean = grouped[["Faith Rate", "Unfaith Rate", "Lengths"]].mean().round(4).reset_index()
    blank_logger.info(tabulate(grouped_mean, headers="keys", tablefmt="pretty", showindex=False))
    
    grouped = df.groupby(["Stage", "Type", "Iter"])
    grouped_mean = grouped[["Faith Rate", "Unfaith Rate", "Lengths"]].mean().round(4).reset_index()
    blank_logger.info(tabulate(grouped_mean, headers="keys", tablefmt="pretty", showindex=False))