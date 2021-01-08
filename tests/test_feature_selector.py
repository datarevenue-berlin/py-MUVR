from unittest.mock import Mock
import pytest
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LinearRegression
from loky import get_reusable_executor
from dask.distributed import Client

from omigami.data import InputDataset, SelectedFeatures, DataSplitter
from omigami.feature_selector import FeatureSelector


@pytest.fixture()
def fs():
    lr = LinearRegression()
    fs = FeatureSelector(
        n_outer=8, n_repetitions=8, random_state=0, estimator=lr, metric="MISS"
    )
    return fs


def test_feature_selector():
    fs = FeatureSelector(
        n_outer=8,
        metric="MISS",
        estimator="RFC",
        features_dropout_rate=0.05,
        robust_minimum=0.05,
        n_repetitions=8,
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
    predefined_group = [1, 2, 3]
    generated_groups = fs.get_groups(None, 5)

    assert all(generated_groups == [0, 1, 2, 3, 4])
    assert fs.get_groups(predefined_group, 5) == predefined_group


def test_run_outer_loop(fs):
    x = np.array([[2, 3, 4], [2, 3, 4]])
    input_data = InputDataset(x, "y", "groups")
    data_splitter = Mock(DataSplitter)
    data_splitter.split_data = Mock(spec=data_splitter.split_data, return_value="split")

    data_splitter.iter_inner_splits = Mock(
        data_splitter.iter_inner_splits, return_value=[1, 2]
    )

    fs._remove_features = Mock(return_value=[])
    fs.create_outer_loop_results = Mock(
        spec=fs.create_outer_loop_results, return_value="outer_loop_res"
    )
    fs.feature_evaluator.evaluate_features = Mock(
        spec=fs.feature_evaluator.evaluate_features, return_value="res"
    )
    fs.post_processor.process_feature_elim_results = Mock(
        fs.post_processor.process_feature_elim_results, return_value="processed_results"
    )
    inner_results = ["res", "res"]
    features = [0, 1, 2]
    raw_results = {tuple(features): inner_results}

    olr = fs._run_outer_loop(input_data, "outer_split", data_splitter)

    assert olr == "outer_loop_res"
    data_splitter.iter_inner_splits.assert_called_with("outer_split")
    data_splitter.split_data.assert_called_with(input_data, 2, features)
    fs._remove_features.assert_called_once_with(features, inner_results)
    fs.feature_evaluator.evaluate_features.assert_called_with("split", features)
    fs.post_processor.process_feature_elim_results.assert_called_with(raw_results)
    fs.create_outer_loop_results.assert_called_with(
        "processed_results", input_data, "outer_split", data_splitter
    )


def test_remove_features(fs, inner_loop_results):
    features = [1, 2, 3, 4]
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
    data_splitter = Mock(DataSplitter)
    data_splitter.split_data = Mock(spec=data_splitter.split_data, return_value="data")

    res = fs.evaluate_min_mid_and_max_features(
        dataset, best_features, "split", data_splitter
    )

    assert res == ("min", "mid", "max")
    data_splitter.split_data.assert_called_with(
        dataset, "split", best_features.max_feats
    )
    assert fs.feature_evaluator.evaluate_features.call_count == 3


@pytest.mark.parametrize(
    "executor",
    [ProcessPoolExecutor(), Client().get_executor(), get_reusable_executor()],
)
def test_deferred_fit(executor):
    X = np.random.rand(10, 10)
    y = np.array([np.random.choice([0, 1]) for _ in range(10)])
    lr = LinearRegression()
    fs = FeatureSelector(
        n_outer=8, n_repetitions=8, random_state=0, estimator=lr, metric="MISS",
    )
    fitted_fs = fs.fit(X, y, executor=executor)
    assert fitted_fs is fs
    assert fs.selected_features
    assert fs.is_fit


def test_select_best_features(fs):
    fs.post_processor.fetch_results = Mock(fs.post_processor.fetch_results, return_value="results")
    fs.post_processor.select_features = Mock(fs.post_processor.select_features, return_value="features")

    selected_features = fs._select_best_features("rep results")

    assert selected_features == "features"
    fs.post_processor.fetch_results.assert_called_once_with("rep results")
    fs.post_processor.select_features.assert_called_once_with("results")


def test_get_selected_features(fs, mosquito):
    X = mosquito.X[:, 0:10]
    y = np.array([1] + [0, 1] * 14)
    fs.fit(X, y)
    selected_features = fs.get_selected_features()
    assert selected_features.min_feats == fs._selected_features.min_feats
    assert selected_features.min_feats == [0]
    feature_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L"]
    assert len(feature_names) == X.shape[1]
    selected_features_names = fs.get_selected_features(feature_names=feature_names)
    assert selected_features_names.min_feats == ["A"]
