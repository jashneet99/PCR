import json
import random
import pandas as pd

random.seed(42)


def format_comve(split: str):
    in_path = f"data/raw/comve/{split}.csv"
    out_path = f"data/formatted/comve/{split}.json"

    df = pd.read_csv(in_path)
    out_data = []

    for idx, row in df.iterrows():
        if random.random() > 0.5:
            sentence0, sentence1 = row['Correct Statement'], row['Incorrect Statement']
            answer, label = 'sentence1', 1
        else:
            sentence0, sentence1 = row['Incorrect Statement'], row['Correct Statement']
            answer, label = 'sentence0', 0

        out_data.append({
            "idx": idx,
            "sentence0": sentence0,
            "sentence1": sentence1,
            "choices": ["sentence0", "sentence1"],
            "gold_answer": answer,
            "gold_label": label,
            "gold_explanation": row['Right Reason1']
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4, ensure_ascii=False)
    print(f"Formatted ComVE ({split}) → {out_path}")
