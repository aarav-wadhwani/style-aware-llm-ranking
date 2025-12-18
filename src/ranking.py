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


from .bootstrap import get_bootstrapped_score, assign_rank

def get_sc_category_results(
    df,
    selected_models,
    style_feature_cols,
    filter_mask=None,
    category_name="Overall w/ Style Control",
    n_bootstrap=10
):
    # optional filter
    if filter_mask is not None:
        df = df[filter_mask].copy()

    pw_df = make_pairwise_feature_df(df, selected_models, style_feature_cols)

    X_id = np.stack(pw_df["X"].values)
    X_style = pw_df[style_feature_cols].to_numpy(dtype=float)
    X_with_style = np.concatenate([X_id, X_style], axis=1)

    y = pw_df["y"].to_numpy(dtype=int)

    feature_labels = list(selected_models) + list(style_feature_cols)

    results_df, _, _ = get_bootstrapped_score(
        X_with_style, y, feature_labels, category_name=category_name, n_bootstrap=n_bootstrap
    )

    results_df["Rank"] = results_df.apply(lambda r: assign_rank(r, results_df), axis=1)

    # style features should not be treated as "models" in the leaderboard view
    results_df.loc[results_df["Model"].isin(style_feature_cols), "Rank"] = -1

    return results_df.sort_values("Rank", ascending=True).reset_index(drop=True)
