import numpy as np
import pandas as pd
from datasets import load_dataset

from src.style_features import add_style_features
from src.ranking import get_sc_category_results

STYLE_COLS = [
    "style_bold_count",
    "style_header_count",
    "style_list_count",
    "style_sum_assistant_tokens",
]

def top_models_by_battle_count(battles, k=20):
    c = battles["model_a"].value_counts().add(battles["model_b"].value_counts(), fill_value=0)
    return list(c.sort_values(ascending=False).head(k).index)

def subselect_battles(battles, selected_models):
    sel = battles[battles["model_a"].isin(selected_models) & battles["model_b"].isin(selected_models)].copy()
    sel_no_ties = sel[sel["winner"] != "tie"].copy()
    return sel, sel_no_ties

def main():
    ds = load_dataset("lmarena-ai/arena-human-preference-100k")
    battles = ds["train"].to_pandas()

    selected_models = top_models_by_battle_count(battles, k=20)
    _, selected_battles_no_ties = subselect_battles(battles, set(selected_models))

    # dedup like your notebook
    selected_battles_no_ties = selected_battles_no_ties[
        selected_battles_no_ties["dedup_tag"].apply(lambda x: x.get("sampled", False))
    ].copy()

    selected_battles_no_ties = add_style_features(selected_battles_no_ties)

    results_df_sc = get_sc_category_results(
        selected_battles_no_ties,
        selected_models,
        STYLE_COLS,
        category_name="Overall w/ Style Control",
        n_bootstrap=10
    )

    print(results_df_sc.head(25))

if __name__ == "__main__":
    main()
