import numpy as np

def turn_into_features(df, models):
    model_to_idx = {m: i for i, m in enumerate(models)}
    X, y = [], []

    for _, row in df.iterrows():
        a, b = row["model_a"], row["model_b"]
        winner = row["winner"]

        xa = np.zeros(len(models))
        xa[model_to_idx[a]] = 1
        xa[model_to_idx[b]] = -1
        ya = 1 if winner == "model_a" else 0

        xb = -xa
        yb = 1 - ya

        X.append(xa)
        y.append(ya)
        X.append(xb)
        y.append(yb)

    return np.array(X), np.array(y)
