import json
import pandas as pd

label_mapping = {'contradiction': 0, 'neutral': 1, 'entailment': 2}


def format_esnli(split: str):
    in_path = f"data/raw/esnli/{split}.csv"
    out_path = f"data/formatted/esnli/{split}.json"

    df = pd.read_csv(in_path)
    out_data = []

    for idx, row in df.iterrows():
        out_data.append({
            'idx': idx,
            'premise': row['Sentence1'],
            'hypothesis': row['Sentence2'],
            'choices': ['contradiction', 'neutral', 'entailment'],
            'gold_answer': row['gold_label'],
            'gold_label': label_mapping[row['gold_label']],
            'gold_explanation': row['Explanation_1']
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4, ensure_ascii=False)
    print(f"Formatted e-SNLI ({split}) → {out_path}")
