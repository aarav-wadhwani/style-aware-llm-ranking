import numpy as np
import pandas as pd

def make_pairwise_feature_df(df, models, style_feature_cols):
    model_to_idx = {m: i for i, m in enumerate(models)}
    records = []

    for _, row in df.iterrows():
        a, b = row["model_a"], row["model_b"]
        winner = row["winner"]
        qid = row.get("question_id", None)

        x = np.zeros(len(models), dtype=float)
        x[model_to_idx[a]] = 1.0
        x[model_to_idx[b]] = -1.0

        y_ab = 1 if winner == "model_a" else 0
        y_ba = 1 - y_ab

        style_ab = {c: float(row[c]) for c in style_feature_cols}
        style_ba = {c: -float(row[c]) for c in style_feature_cols}

        rec_ab = {"question_id": qid, "X": x.tolist(), "y": y_ab, "direction": "A->B", **style_ab}
        rec_ba = {"question_id": qid, "X": (-x).tolist(), "y": y_ba, "direction": "B->A", **style_ba}

        records.append(rec_ab)
        records.append(rec_ba)

    cols = ["question_id", "X", "y", "direction"] + style_feature_cols
    return pd.DataFrame(records)[cols]
