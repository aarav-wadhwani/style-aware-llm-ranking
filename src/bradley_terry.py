from sklearn.linear_model import LogisticRegression

def fit_bt_model(X, y):
    model = LogisticRegression(
        fit_intercept=False,
        max_iter=1000,
        solver="lbfgs"
    )
    model.fit(X, y)
    return model.coef_[0]
