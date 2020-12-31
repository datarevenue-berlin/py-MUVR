import pytest
from concurrent.futures import ProcessPoolExecutor
from omigami.feature_selector import FeatureSelector
from omigami.outer_loop import OuterLoop
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
    fs = FeatureSelector(
        n_outer=8, repetitions=8, random_state=0, estimator=lr, metric="MISS"
    )
    fitted_fs = fs.fit(X, y)
    assert fitted_fs is fs
    assert fs.selected_features


def test_deferred_fit():
    X = np.random.rand(10, 10)
    y = np.array([np.random.choice([0, 1]) for _ in range(10)])
    lr = LinearRegression()
    fs = FeatureSelector(
        n_outer=8,
        repetitions=8,
        random_state=0,
        estimator=lr,
        metric="MISS",
        executor=ProcessPoolExecutor(),
    )
    fitted_fs = fs.fit(X, y)
    assert fitted_fs is fs
    assert fs.selected_features


def test_execute_repetitions():
    class MockOuterLoop:
        refresh_count = 0
        run_count = 0

        def refresh_splits(self):
            self.refresh_count += 1

        def run(self, executor=None):
            self.run_count += 1
            return self.run_count

    outer_loop = MockOuterLoop()
    fs = FeatureSelector(n_outer=8, repetitions=8, estimator="RFC", metric="MISS")
    reps = fs._execute_repetitions(outer_loop)
    assert len(reps) == 8
    assert outer_loop.refresh_count == 8
    assert outer_loop.run_count == 8
    assert sorted(reps) == [1, 2, 3, 4, 5, 6, 7, 8]

