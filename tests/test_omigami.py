import collections
import pytest
import pandas as pd
from sklearn import datasets
from omigami.feature_selector import FeatureSelector

Dataset = collections.namedtuple("Dataset", "X y groups")


@pytest.fixture
def mosquito():
    df = pd.read_csv("tests/assets/mosquito.csv").set_index("Unnamed: 0")
    X = df.drop(columns=["Yotu"]).values
    y = df.Yotu.values
    groups = df.index
    return Dataset(X=X, y=y, groups=groups)


@pytest.fixture
def feature_selector_mosquito(mosquito):
    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="RFC",)
    fs.n_features = mosquito.X.shape[1]
    return fs


@pytest.fixture
def feature_selector():
    return FeatureSelector(
        n_outer=2, n_inner=2, repetitions=1, estimator="RFC", metric="MISS"
    )


@pytest.fixture
def outer_loop_aggregation():
    return [
        {
            "avg_feature_ranks": {
                "min": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
                "max": {0: 4, 1: 2, 2: 3, 3: 4, 4: 5},
                "mid": {0: 4, 1: 3, 2: 3, 3: 2, 4: 1},
            },
            "scores": [{5: 5, 4: 4, 3: 4, 2: 4}, {5: 10, 4: 9, 3: 10, 2: 11}],
            "n_feats": {"min": 4, "max": 4, "mid": 4},
        },
        {
            "avg_feature_ranks": {
                "min": {0: 3, 1: 1, 2: 5, 3: 5, 4: 5},
                "max": {0: 1, 1: 2, 2: 3, 3: 5, 4: 5},
                "mid": {0: 1, 1: 3, 2: 2, 3: 5, 4: 5},
            },
            "scores": [{5: 5, 4: 3, 3: 3, 2: 4}, {5: 10, 4: 10, 3: 10, 2: 12}],
            "n_feats": {"min": 3, "max": 4, "mid": 3},
        },
    ]


def test_feature_selector_creation(feature_selector_mosquito):
    assert feature_selector_mosquito
    assert feature_selector_mosquito.n_inner == 4
    assert feature_selector_mosquito.features_dropout_rate == 0.05


def test_fit(feature_selector):
    X, y = datasets.make_classification(n_features=5, random_state=0)
    assert feature_selector.fit(X, y)
    assert feature_selector.n_features == X.shape[1]


def test_process_results(feature_selector, results):
    feature_selector.n_features = 5  # to mimick fit
    best_feats = feature_selector._process_results(results)
    assert best_feats["min"]
    assert best_feats["max"]
    assert best_feats["mid"]
    assert best_feats["min"] == {0, 1}
    assert best_feats["max"] == {0, 1, 3, 4}
    assert best_feats["mid"] == {0, 1, 4}


def test_process_outer_loop(feature_selector, results):
    feature_selector.n_features = 5  # to mimick fit
    processed = feature_selector._process_outer_loop(results)
    assert processed["avg_feature_ranks"]
    assert processed["scores"]
    assert processed["n_feats"]
    assert processed["n_feats"]["min"] == 2
    assert processed["n_feats"]["max"] == 2
    assert processed["n_feats"]["mid"] == 2


def test_compute_avg_feature_rank(feature_selector_mosquito, results):
    avg_rks = feature_selector_mosquito._compute_avg_feature_rank(results)
    assert avg_rks["min"]
    assert avg_rks["max"]
    assert avg_rks["mid"]
    avg_rks = avg_rks["min"]
    assert avg_rks[0] == 2.5
    assert avg_rks[1] == 2.5
    assert avg_rks[2] == 2.5
    assert avg_rks[3] == 4.5
    assert avg_rks[4] == (5 + feature_selector_mosquito.n_features) / 2


def test_select_best_features(feature_selector_mosquito, outer_loop_aggregation):
    feature_sets = feature_selector_mosquito._select_best_features(
        outer_loop_aggregation
    )
    assert feature_sets
    assert feature_sets["min"] == {0, 1, 2, 3}
    assert feature_sets["max"] == {0, 1, 2, 3}
    assert feature_sets["mid"] == {0, 1, 2, 4}


def test_compute_final_ranks(feature_selector_mosquito, outer_loop_aggregation):
    ranks = feature_selector_mosquito._compute_final_ranks(outer_loop_aggregation)
    assert not ranks.empty
    assert ranks["min"].values.all()
    assert ranks["max"].values.all()
    assert ranks["mid"].values.all()
    assert len(ranks) == 5
    assert ranks.loc[0, "min"] == 2
    assert ranks.loc[4, "mid"] == 3
