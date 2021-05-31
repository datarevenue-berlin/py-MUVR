from unittest.mock import Mock
import pytest
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from loky import get_reusable_executor
from dask.distributed import Client

from py_muvr.data_structures import (
    InputDataset,
    SelectedFeatures,
    FeatureSelectionResults,
    ScoreCurve,
)
from py_muvr.data_splitter import DataSplitter
from py_muvr.feature_selector import FeatureSelector


@pytest.fixture()
def fs():
    lr = LinearRegression()
    fs = FeatureSelector(
        n_outer=4,
        n_repetitions=4,
        random_state=0,
        estimator=lr,
        metric="MISS",
        features_dropout_rate=0.1,
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
    assert fs._selected_features
    assert fs.is_fit


def test_get_groups(fs):
    predefined_group = [1, 2, 3]
    generated_groups = fs._get_groups(None, 5)

    assert all(generated_groups == [0, 1, 2, 3, 4])
    assert fs._get_groups(predefined_group, 5) == predefined_group


def test_run_outer_loop(fs):
    x = np.array([[2, 3, 4], [2, 3, 4]])
    input_data = InputDataset(x, "y", "groups")
    data_splitter = Mock(DataSplitter)
    data_splitter.split_data = Mock(spec=data_splitter.split_data, return_value="split")

    data_splitter.iter_inner_splits = Mock(
        data_splitter.iter_inner_splits, return_value=[1, 2]
    )

    fs._remove_features = Mock(return_value=[])
    fs._create_outer_loop_results = Mock(
        spec=fs._create_outer_loop_results, return_value="outer_loop_res"
    )
    fs._feature_evaluator.evaluate_features = Mock(
        spec=fs._feature_evaluator.evaluate_features, return_value="res"
    )
    fs._post_processor.process_feature_elim_results = Mock(
        fs._post_processor.process_feature_elim_results,
        return_value="processed_results",
    )
    inner_results = ["res", "res"]
    features = [0, 1, 2]
    raw_results = {tuple(features): inner_results}

    olr = fs._run_outer_loop(input_data, "outer_split", data_splitter)

    assert olr == "outer_loop_res"
    data_splitter.iter_inner_splits.assert_called_with("outer_split")
    data_splitter.split_data.assert_called_with(input_data, 2, features)
    fs._remove_features.assert_called_once_with(features, inner_results)
    fs._feature_evaluator.evaluate_features.assert_called_with("split", features)
    fs._create_outer_loop_results.assert_called_with(
        raw_results, input_data, "outer_split", data_splitter
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
    fs._feature_evaluator.evaluate_features = Mock(
        spec=fs._feature_evaluator.evaluate_features, side_effect=["min", "mid", "max"]
    )
    data_splitter = Mock(DataSplitter)
    data_splitter.split_data = Mock(spec=data_splitter.split_data, return_value="data")

    res = fs._evaluate_min_mid_and_max_features(
        dataset, best_features, "split", data_splitter
    )

    assert res == ("min", "mid", "max")
    data_splitter.split_data.assert_called_with(dataset, "split", best_features["max"])
    assert fs._feature_evaluator.evaluate_features.call_count == 3


@pytest.mark.parametrize(
    "executor",
    [ProcessPoolExecutor(), Client().get_executor(), get_reusable_executor()],
)
def test_deferred_fit(executor):
    X = np.random.rand(10, 10)
    y = np.array([np.random.choice([0, 1]) for _ in range(10)])
    lr = LinearRegression()
    fs = FeatureSelector(
        n_outer=3,
        n_repetitions=2,
        random_state=0,
        estimator=lr,
        metric="MISS",
    )
    fitted_fs = fs.fit(X, y, executor=executor)
    assert fitted_fs is fs
    assert fs._selected_features
    assert fs.is_fit


def test_select_best_features(fs):
    fs._fetch_results = Mock(fs._fetch_results, return_value="results")
    fs._post_processor.select_features = Mock(
        fs._post_processor.select_features, return_value="features"
    )

    selected_features = fs._select_best_features("rep results")

    assert selected_features == "features"
    fs._fetch_results.assert_called_once_with("rep results")
    fs._post_processor.select_features.assert_called_once_with("results")


def test_get_feature_selection_results(fs, raw_results):
    fs._selected_features = [1, 2, 3]
    fs._raw_results = raw_results
    fs.is_fit = True
    fs._get_selected_feature_names = Mock(
        fs._get_selected_feature_names, return_value="sel_feat_names"
    )
    fs._get_validation_curves = Mock(
        fs._get_validation_curves, return_value={"total": [ScoreCurve(None, None)]}
    )

    fs_results = fs.get_feature_selection_results(["names"])

    assert isinstance(fs_results, FeatureSelectionResults)
    assert isinstance(fs_results.score_curves["total"][0], ScoreCurve)
    assert fs_results.selected_features == fs._selected_features
    assert fs_results.selected_features is not fs._selected_features
    assert fs_results.raw_results == fs._raw_results
    assert fs_results.raw_results is not fs._raw_results
    assert fs_results.selected_feature_names == "sel_feat_names"
    fs._get_selected_feature_names.assert_called_once_with(["names"])
    fs._get_validation_curves.assert_called_once()


def test_get_selected_features(fs):
    fs._selected_features = [1, 2, 3]

    selected_features = fs.get_selected_features()

    assert selected_features == fs._selected_features
    assert selected_features is not fs._selected_features


def test_get_selected_feature_names(fs, mosquito):
    X = mosquito.X[:, 0:10]
    y = np.array([1] + [0, 1] * 14)
    fs.fit(X, y)
    fs._selected_features = SelectedFeatures([0], [0], [0])
    selected_features = fs._selected_features
    feature_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L"]

    selected_features_names = fs._get_selected_feature_names(
        feature_names=feature_names
    )

    assert len(selected_features_names["min"]) == len(selected_features["min"])
    with pytest.raises(ValueError):
        fs._get_selected_feature_names(feature_names=["only-one-name"])


def test_export_average_feature_ranks(fs):
    df = pd.DataFrame()
    df.to_csv = Mock()
    fs.get_average_ranks_df = Mock(return_value=df)

    res = fs.export_average_feature_ranks("path", ["names"], True)

    assert res is df
    fs.get_average_ranks_df.assert_called_once_with(["names"], True)
    df.to_csv.assert_called_once_with("path")


def test_get_average_ranks_df(fs):
    fs.get_feature_selection_results = Mock(return_value="fs_results")
    fs._post_processor.make_average_ranks_df = Mock(return_value="ranks_df")

    ranks_df = fs.get_average_ranks_df(["names"], True)

    assert ranks_df == "ranks_df"
    fs.get_feature_selection_results.assert_called_once()
    fs._post_processor.make_average_ranks_df.assert_called_once_with(
        "fs_results", fs._n_features, ["names"], True
    )
