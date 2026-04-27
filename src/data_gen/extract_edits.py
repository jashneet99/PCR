import os
import re
import json
import argparse
from tqdm import tqdm

def extract(generated_edits):
    lines = generated_edits.strip().split('\n')

    edit_list = []
    word_list = []

    for line in lines:
        match = re.match(r"(\d+)\.\s(.+)", line)
        if match:
            edit_content = match.group(2)
            word_match = re.search(r"\[(.*?)\]", edit_content)
            if word_match:
                word = word_match.group(1)
                edit_content_clean = re.sub(r"\[(.*?)\]", word, edit_content).strip()
                edit_list.append(edit_content_clean)
                word_list.append(word)

    return edit_list, word_list


def extract_edits(dataset, dataset_name):
    extract_edits_dataset = []
    extract_edits_failed = []
    extract_edits_failed_idx = []

    if dataset_name == 'esnli':
        for data in tqdm(dataset, desc="Processing"):
            premise = data['premise']
            hypothesis = data['hypothesis']

            premise_generated_edits = data['edit_gen_premise']
            hypothesis_generated_edits = data['edit_gen_hypothesis']

            premise_edit_list, premise_word_list = extract(premise_generated_edits)
            hypothesis_edit_list, hypothesis_word_list = extract(hypothesis_generated_edits)

            if len(premise_edit_list) != len(hypothesis_edit_list):
                extract_edits_failed.append(data)
                extract_edits_failed_idx.append(data['idx'])
                continue

            eidx = 0
            for i, premise_edit in enumerate(premise_edit_list):
                extract_edits_dataset.append({
                    'idx': data['idx'],
                    'eidx': eidx,
                    'premise': premise_edit,
                    'hypothesis': hypothesis,
                    'choices': data['choices'],
                    'edit_pos': 'premise',
                    'edit_word': premise_word_list[i],
                })
                eidx += 1
            for j, hypothesis_edit in enumerate(hypothesis_edit_list):
                extract_edits_dataset.append({
                    'idx': data['idx'],
                    'eidx': eidx,
                    'premise': premise,
                    'hypothesis': hypothesis_edit,
                    'choices': data['choices'],
                    'edit_pos': 'hypothesis',
                    'edit_word': hypothesis_word_list[j],
                })
                eidx += 1

    elif dataset_name == 'comve':
        for data in tqdm(dataset, desc="Processing"):
            gold_answer = data['gold_answer']  # 'sentence0' or 'sentence1'

            # The correct (unchanged) sentence is the opposite one
            correct_answer = 'sentence1' if gold_answer == 'sentence0' else 'sentence0'

            wrong_sentence = data[gold_answer]
            correct_sentence = data[correct_answer]

            # Only the wrong sentence has edits generated
            generated_edits_key = f'edit_gen_{gold_answer}'

            if generated_edits_key not in data:
                extract_edits_failed.append(data)
                extract_edits_failed_idx.append(data['idx'])
                continue

            edit_list, word_list = extract(data[generated_edits_key])

            if len(edit_list) == 0:
                extract_edits_failed.append(data)
                extract_edits_failed_idx.append(data['idx'])
                continue

            eidx = 0
            for i, wrong_sentence_edit in enumerate(edit_list):
                if gold_answer == 'sentence0':
                    # edited sentence0 paired with unchanged sentence1
                    extract_edits_dataset.append({
                        'idx': data['idx'],
                        'eidx': eidx,
                        'sentence0': wrong_sentence_edit,
                        'sentence1': correct_sentence,
                        'choices': data['choices'],
                        'gold_answer': data['gold_answer'],
                        'gold_label': data['gold_label'],
                        'gold_explanation': data['gold_explanation'],
                        'edit_pos': gold_answer,
                        'edit_word': word_list[i],
                    })
                else:
                    # edited sentence1 paired with unchanged sentence0
                    extract_edits_dataset.append({
                        'idx': data['idx'],
                        'eidx': eidx,
                        'sentence0': correct_sentence,
                        'sentence1': wrong_sentence_edit,
                        'choices': data['choices'],
                        'gold_answer': data['gold_answer'],
                        'gold_label': data['gold_label'],
                        'gold_explanation': data['gold_explanation'],
                        'edit_pos': gold_answer,
                        'edit_word': word_list[i],
                    })
                eidx += 1

    elif dataset_name == 'ecqa':
        for data in tqdm(dataset, desc="Processing"):
            question_generated_edits = data['edit_gen_question']
            question_edit_list, question_word_list = extract(question_generated_edits)
            question_edit_list = question_edit_list[:10]
            question_word_list = question_word_list[:10]

            if len(question_edit_list) != 10:
                extract_edits_failed.append(data)
                extract_edits_failed_idx.append(data['idx'])
                continue

            eidx = 0
            for i, question_edit in enumerate(question_edit_list):
                extract_edits_dataset.append({
                    'idx': data['idx'],
                    'eidx': eidx,
                    'question': question_edit,
                    'choices': data['choices'],
                    'edit_pos': 'question',
                    'edit_word': question_word_list[i],
                })
                eidx += 1

    return extract_edits_dataset, extract_edits_failed, extract_edits_failed_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', "--dir", type=str, default='data')
    parser.add_argument('-d', '--dataset', type=str, default='comve', choices=['comve', 'ecqa', 'esnli'])
    parser.add_argument('-dt', '--dataset_type', type=str, default='org', choices=['org', 'failed', 'final'])
    args = parser.parse_args()

    dir = args.dir
    dataset_name = args.dataset
    dataset_type = args.dataset_type

    edits_gen_path = os.path.join(dir, 'counterfactual', dataset_name, f"gen_{dataset_type}.json")
    with open(edits_gen_path, 'r') as f:
        edits_gen = json.load(f)

    extract_edits_dataset, extract_edits_failed, extract_edits_failed_idx = extract_edits(edits_gen, dataset_name)

    edits_ext = {
        'total': len(extract_edits_dataset),
        'num_failed': len(extract_edits_failed_idx),
        'extract_edits_failed_idx': extract_edits_failed_idx,
        'extract_edits_failed': extract_edits_failed,
        'extract_edits_dataset': extract_edits_dataset
    }

    edits_ext_path = os.path.join(dir, 'counterfactual', dataset_name, f"ext_{dataset_type}.json")

    output_dir = os.path.dirname(edits_ext_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(edits_ext_path, 'w') as f:
        json.dump(edits_ext, f, indent=4)

    print(f"Total extracted: {len(extract_edits_dataset)}")
    print(f"Failed: {len(extract_edits_failed_idx)} → idx: {extract_edits_failed_idx}")

main()