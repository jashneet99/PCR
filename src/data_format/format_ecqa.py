import json
import pandas as pd


def format_ecqa(split: str):
    in_path = f"data/raw/ecqa/{split}.csv"
    out_path = f"data/formatted/ecqa/{split}.json"

    df = pd.read_csv(in_path)
    out_data = []

    for idx, row in df.iterrows():
        choices = [row[f"q_op{i}"] for i in range(1, 6)]
        label = choices.index(row['q_ans']) if row['q_ans'] in choices else -1

        out_data.append({
            "idx": idx,
            "question": row['q_text'],
            "choices": choices,
            "gold_answer": row['q_ans'],
            "gold_label": label,
            "gold_explanation": row['taskB']
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4, ensure_ascii=False)
    print(f"Formatted ECQA ({split}) → {out_path}")
