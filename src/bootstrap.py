import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def get_bootstrapped_score(X, y, feature_labels, category_name="Overall", n_bootstrap=10, seed=189):
    """
    Bootstraps logistic regression coefficients to estimate confidence intervals.
    Returns:
      results_df: Model/Feature, Average Score, Lower Bound, Upper Bound, Category
      mean_scores: np.array
      confidence_intervals: np.array shape [2, n_features]
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(X), size=len(X), replace=True)
        Xs, ys = X[idx], y[idx]

        # guard: bootstrap sample must contain both classes
        if len(np.unique(ys)) < 2:
            continue

        m = LogisticRegression(fit_intercept=False, max_iter=1000)
        m.fit(Xs, ys)
        bootstrap_scores.append(m.coef_[0])

    if len(bootstrap_scores) == 0:
        raise ValueError("All bootstrap samples collapsed to a single class. Increase n_bootstrap or check labels.")

    bootstrap_scores = np.array(bootstrap_scores)
    mean_scores = bootstrap_scores.mean(axis=0)
    confidence_intervals = np.percentile(bootstrap_scores, [2.5, 97.5], axis=0)

    results_df = pd.DataFrame({
        "Model": feature_labels,
        "Average Score": mean_scores,
        "Lower Bound": confidence_intervals[0],
        "Upper Bound": confidence_intervals[1],
        "Category": category_name,
    }).sort_values("Average Score", ascending=False).reset_index(drop=True)

    return results_df, mean_scores, confidence_intervals


def assign_rank(row, df):
    """
    Rank = (# of models/features confidently better) + 1.
    Confidently better means: other.lower_bound > current.upper_bound
    """
    return int((df["Lower Bound"] > row["Upper Bound"]).sum() + 1)
