import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from model.model import GenerationModel


def generate(model, prompt):
    outputs = model.get_generated(prompt, do_sample=False, max_new_tokens=512)
    return outputs[0]


def generate_edits(dataset, dataset_name, model):
    edits_dataset = []

    if dataset_name in ['esnli', 'comve']:
        current_dir = os.path.dirname(__file__)
        prompt_prefix_path = os.path.join(current_dir, 'edit_prompt_10.txt')
        with open(prompt_prefix_path, 'r') as f:
            prompt_prefix = f.read()

        if dataset_name == 'esnli':
            for data in tqdm(dataset, desc="Processing"):
                premise = data['premise']
                hypothesis = data['hypothesis']
                premise_prompt = prompt_prefix + premise + "\nOutput:\n"
                hypothesis_prompt = prompt_prefix + hypothesis + "\nOutput:\n"
                data['edit_gen_premise'] = generate(model, premise_prompt)
                data['edit_gen_hypothesis'] = generate(model, hypothesis_prompt)
                edits_dataset.append(data)

        elif dataset_name == 'comve':
            for data in tqdm(dataset, desc="Processing"):
                gold_answer = data['gold_answer']
                wrong_sentence = data[gold_answer]
                wrong_sentence_prompt = prompt_prefix + wrong_sentence + "\nOutput:\n"
                generated_edits = generate(model, wrong_sentence_prompt)
                data[f'edit_gen_{gold_answer}'] = generated_edits
                edits_dataset.append(data)

    elif dataset_name in ['ecqa']:
        current_dir = os.path.dirname(__file__)
        prompt_prefix_path = os.path.join(current_dir, 'edit_prompt_20.txt')
        with open(prompt_prefix_path, 'r') as f:
            prompt_prefix = f.read()

        for data in tqdm(dataset, desc="Processing"):
            question = data['question']
            question_prompt = prompt_prefix + question + "\nOutput:\n"
            data['edit_gen_question'] = generate(model, question_prompt)
            edits_dataset.append(data)

    return edits_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', "--dir", type=str, default='data')
    parser.add_argument('-d', '--dataset', type=str, default='comve', choices=['comve', 'ecqa', 'esnli'])
    parser.add_argument('-dt', '--dataset_type', type=str, default='org', choices=['org', 'failed'])
    parser.add_argument('-n', '--num_data', type=int, default=None)
    parser.add_argument('-m', '--model_name', type=str, default='llama',
                        choices=['llama', 'mistral', 'qwen', 'falcon'])
    args = parser.parse_args()

    dir = args.dir
    dataset_name = args.dataset
    dataset_type = args.dataset_type
    num_data = args.num_data
    model_name = args.model_name

    print(f"Loading model: {model_name}...")
    model = GenerationModel(model_name)
    print("Model loaded!")

    if dataset_type == 'org':
        dataset_path = os.path.join('data/formatted', dataset_name, 'test.json')
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        if num_data is not None:
            dataset = dataset[:num_data]

    elif dataset_type == 'failed':
        dataset_path = os.path.join(dir, 'counterfactual', dataset_name, "ext_org.json")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)['extract_edits_failed']

    print(f"Processing {len(dataset)} records from {dataset_path}")

    edits_dataset = generate_edits(dataset, dataset_name, model)

    if dataset_type == 'org':
        edits_output_path = os.path.join(dir, 'counterfactual', dataset_name, "gen_org.json")
    elif dataset_type == 'failed':
        edits_output_path = os.path.join(dir, 'counterfactual', dataset_name, "gen_failed.json")

    output_dir = os.path.dirname(edits_output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(edits_output_path, 'w') as f:
        json.dump(edits_dataset, f, indent=4)

    print(f"Saved {len(edits_dataset)} records to {edits_output_path}")

main()
