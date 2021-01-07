from unittest.mock import Mock

import pytest
from concurrent.futures import ProcessPoolExecutor

from data_models import InputData, SelectedFeatures
from data_splitter import DataSplitter
from feature_selector import FeatureSelector
import numpy as np
from sklearn.linear_model import LinearRegression

from post_processor import PostProcessor


@pytest.fixture()
def fs():
    lr = LinearRegression()
    fs = FeatureSelector(
        n_outer=8, repetitions=8, random_state=0, estimator=lr, metric="MISS"
    )
    return fs


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
    x = np.array([[2, 3]])
    input_data = InputData(x, "y", "groups", 3)
    input_data.split_data = Mock(spec=input_data.split_data, return_value="split")

    data_splitter = Mock(spec=DataSplitter)
    data_splitter.iter_inner_splits = Mock(data_splitter.iter_inner_splits, return_value=[1, 2])

    fs._remove_features = Mock(return_value=[])
    fs.create_outer_loop_results = Mock(
        spec=fs.create_outer_loop_results, return_value="outer_loop_res"
    )
    fs.feature_evaluator.evaluate_features = Mock(
        spec=fs.feature_evaluator.evaluate_features, return_value="res"
    )
    inner_results = ["res", "res"]
    features = [0, 1, 2]
    feature_elim_results = {tuple(features): inner_results}

    olr = fs._run_outer_loop(input_data, data_splitter, "outer_split")

    assert olr == "outer_loop_res"
    data_splitter.iter_inner_splits.assert_called_with("outer_split")
    input_data.split_data.assert_called_with(2, features)
    fs._remove_features.assert_called_once_with(features, inner_results)
    fs.feature_evaluator.evaluate_features.assert_called_with("split", features)
    fs.create_outer_loop_results.assert_called_with(feature_elim_results, input_data, "outer_split")


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


def test_create_outer_loop_results():
    pass  # TODO


def test_evaluate_min_mid_and_max_features(fs, dataset, rfe_raw_results):
    best_features = SelectedFeatures([1, 2], [1, 2, 3, 4], [1, 2, 3])
    fs.feature_evaluator.evaluate_features = Mock(
        spec=fs.feature_evaluator.evaluate_features, side_effect=["min", "mid", "max"]
    )
    dataset.split_data = Mock(spec=dataset.split_data, return_value="data")

    res = fs.evaluate_min_mid_and_max_features(dataset, best_features, "split")

    assert res == ("min", "mid", "max")
    dataset.split_data.assert_called_with("split")
    assert fs.feature_evaluator.evaluate_features.call_count == 3


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
