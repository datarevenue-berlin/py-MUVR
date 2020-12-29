from omigami.feature_selector import FeatureSelector
import numpy as np
from sklearn.linear_model import LinearRegression


def test_feature_selector():
    fs = FeatureSelector(
        n_outer=8,
        metric="MISS",
        estimator="RFC",
        features_dropout_rate=0.05,
        robust_minimum=0.05,
        repetitions=8,
        random_state=0,
    )
    assert fs
    assert fs.n_inner == 7


def test_fit():
    X = np.random.rand(10, 10)
    y = np.array([np.random.choice([0, 1]) for _ in range(10)])
    lr = LinearRegression()
    fs = FeatureSelector(n_outer=8, repetitions=8, random_state=0, estimator=lr, metric="MISS")
    fitted_fs = fs.fit(X, y)
    assert fitted_fs is fs
    assert fs.selected_features
