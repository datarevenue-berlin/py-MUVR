from omigami.feature_selector import FeatureSelector
import numpy as np


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
    y = np.round(np.random.rand(10))
    fs = FeatureSelector(n_outer=8, repetitions=8, random_state=0)
    assert fs.fit(X, y)
