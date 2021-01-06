from unittest.mock import Mock

import pytest
from concurrent.futures import ProcessPoolExecutor

from data_models import InputData, FeatureEvaluationResults, FeatureRanks
from data_splitter import DataSplitter
from feature_selector import FeatureSelector
import numpy as np
from sklearn.linear_model import LinearRegression


@pytest.fixture()
def fs():
    lr = LinearRegression()
    fs = FeatureSelector(
        n_outer=8, repetitions=8, random_state=0, estimator=lr, metric="MISS"
    )
    return fs


@pytest.fixture
def inner_loop_results():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3, 4], ranks=[3, 2, 1, 4]), test_score=0.2,
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3, 4], ranks=[1.5, 1.5, 3, 4]), test_score=0.2,
        ),
    ]


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


def test_fit(fs):
    X = np.random.rand(10, 10)
    y = np.array([np.random.choice([0, 1]) for _ in range(10)])
    fitted_fs = fs.fit(X, y)
    assert fitted_fs is fs
    assert fs.selected_features
    assert fs.is_fit


def test_get_groups(fs):
    predefined_group = [1, 2 ,3]
    generated_groups = fs.get_groups(None, 5)

    assert all(generated_groups == [0, 1, 2, 3, 4])
    assert fs.get_groups(predefined_group, 5) == predefined_group


def test_run_outer_loop(fs):
    fs.feature_evaluator.evaluate_features = Mock(
        spec=fs.feature_evaluator.evaluate_features, return_value="res"
    )
    input_data = Mock(spec=InputData)
    data_splitter = Mock(spec=DataSplitter)
    fs.compute_outer_loop_results = Mock(
        spec=fs.compute_outer_loop_results, return_value="outer_loop_res"
    )

    olr = fs._run_outer_loop(input_data, data_splitter, "outer_split")

    assert olr == "outer_loop_res"
    assert False


def test_remove_features(fs, inner_loop_results):
    features=np.array([1, 2, 3, 4])
    fs.keep_fraction = 0.75
    features = fs._remove_features(features, inner_loop_results)
    assert len(features) == 3
    assert features == [2, 3, 1]


def test_select_n_best(fs, inner_loop_results):
    keep = 2
    n_best = fs._select_n_best(inner_loop_results, keep)

    assert n_best == [2, 3]


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
    assert fs.is_fit


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


def test_select_best_features(
    inner_loop_results, inner_loop_results_2, feature_evaluator
):
    fs = FeatureSelector(n_outer=8, repetitions=8, estimator="RFC", metric="MISS")
    feature_elim_results = {(1, 2, 3): inner_loop_results, (1,): inner_loop_results_2}
    avg_scores = fs._compute_score_curve(feature_elim_results)
    selected_feats = fs._select_best_features(feature_elim_results, avg_scores)
    # edge case: it's just two recursive steps at 3 and 1 features. The one at
    # 3 is the best (lowest test-score) so every feature set should be [1, 2, 3]
    assert sorted(selected_feats.min_feats) == [1, 2, 3]
    assert sorted(selected_feats.mid_feats) == [1, 2, 3]
    assert sorted(selected_feats.max_feats) == [1, 2, 3]


def test_compute_score_curve(
    rfe, inner_loop_results, inner_loop_results_2,
):
    rfe_res = {(1, 2, 3): inner_loop_results, (1,): inner_loop_results_2}
    avg_scores = rfe._compute_score_curve(rfe_res)
    assert len(avg_scores) == 2
    assert 1 in avg_scores
    assert 3 in avg_scores
    assert avg_scores[3] < avg_scores[1]
