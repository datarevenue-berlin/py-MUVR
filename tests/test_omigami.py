import collections
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, datasets
from omigami.omigami import FeatureSelector, miss_score


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
    return FeatureSelector(
        X=mosquito.X,
        y=mosquito.y,
        groups=mosquito.groups,
        n_outer=5,
        metric="MISS",
        estimator="RFC",
    )


@pytest.fixture
def feature_selector():
    X, y = datasets.make_classification(n_features=5, random_state=0)
    return FeatureSelector(
        X, y, n_outer=2, n_inner=2, repetitions=1, estimator="RFC", metric="MISS"
    )


@pytest.fixture
def results():
    return [
        [
            {
                "test_results": {
                    "min": {"score": 4, "feature_ranks": {0: 1.0, 1: 2.0}},
                    "max": {
                        "score": 5,
                        "feature_ranks": {0: 1.0, 1: 2.0, 2: 4.0, 3: 3.0},
                    },
                    "mid": {"score": 5, "feature_ranks": {0: 1.0, 1: 2.0, 3: 3.0}},
                },
                "scores": {5: 4, 4: 3, 3: 3, 2: 3},
            },
            {
                "test_results": {
                    "min": {
                        "score": 3,
                        "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0, 3: 4.0},
                    },
                    "max": {
                        "score": 3,
                        "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0, 3: 4.0},
                    },
                    "mid": {
                        "score": 2,
                        "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0, 3: 4.0},
                    },
                },
                "scores": {5: 5, 4: 4, 3: 5, 2: 5},
            },
        ],
        [
            {
                "test_results": {
                    "min": {"score": 4, "feature_ranks": {0: 1.0, 1: 2.0}},
                    "max": {
                        "score": 5,
                        "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0, 2: 4.0},
                    },
                    "mid": {"score": 5, "feature_ranks": {0: 2.0, 1: 1.0, 4: 3.0}},
                },
                "scores": {5: 5, 4: 3, 3: 5, 2: 3},
            },
            {
                "test_results": {
                    "min": {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0}},
                    "max": {
                        "score": 2,
                        "feature_ranks": {0: 1.0, 1: 2.0, 2: 5.0, 3: 4.0, 4: 3.0},
                    },
                    "mid": {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0}},
                },
                "scores": {5: 5, 4: 6, 3: 5, 2: 5},
            },
        ],
    ]


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


@pytest.fixture
def outer_loop_results():
    return [
        {
            "test_results": {
                "min": {"score": 4, "feature_ranks": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}},
                "max": {"score": 3, "feature_ranks": {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}},
                "mid": {"score": 5, "feature_ranks": {0: 2, 1: 3, 2: 4, 3: 5, 4: 1}},
            },
            "scores": {5: -5, 4: -5, 3: -4, 2: -4},
        },
        {
            "test_results": {
                "min": {"score": 7, "feature_ranks": {0: 4, 1: 3, 2: 2, 3: 5, 4: 5,}},
                "max": {"score": 6, "feature_ranks": {0: 1, 1: 2, 2: 3, 3: 5, 4: 5}},
                "mid": {"score": 7, "feature_ranks": {0: 3, 1: 2, 2: 2, 3: 5, 4: 5}},
            },
            "scores": {5: -7, 4: -5, 3: -7, 2: -11},
        },
    ]


@pytest.fixture
def inner_results():
    return [
        {"feature_ranks": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}},
        {"feature_ranks": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}},
        {"feature_ranks": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}},
    ]


@pytest.fixture
def outer_train_results():
    return {
        (0, 1, 2, 3, 4): [
            {"score": 1, "feature_ranks": {0: 2.0, 1: 1.0, 2: 4.0, 3: 5.0, 4: 3.0}},
            {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0, 2: 5.0, 3: 4.0, 4: 3.0}},
        ],
        (0, 1, 4, 2): [
            {"score": 1, "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0, 2: 4.0}},
            {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0, 2: 4.0}},
        ],
        (0, 1, 4): [
            {"score": 3, "feature_ranks": {0: 2.0, 1: 1.0, 4: 3.0}},
            {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0, 4: 3.0}},
        ],
        (0, 1): [
            {"score": 1, "feature_ranks": {0: 1.0, 1: 2.0}},
            {"score": 2, "feature_ranks": {0: 1.0, 1: 2.0}},
        ],
    }


@pytest.fixture
def scores():
    return [{5: 5, 4: 4, 3: 4, 2: 4}, {5: 10, 4: 9, 3: 10, 2: 11}]


def test_feature_selector_creation(feature_selector_mosquito):
    assert feature_selector_mosquito
    assert feature_selector_mosquito.n_inner == 4
    assert feature_selector_mosquito.features_dropout_rate == 0.05
    assert feature_selector_mosquito.groups.size


def test_select_features(feature_selector):
    assert feature_selector.select_features()


def test_process_results(feature_selector, results):
    best_feats = feature_selector._process_results(results)
    assert best_feats["min"]
    assert best_feats["max"]
    assert best_feats["mid"]
    assert best_feats["min"] == {0, 1}
    assert best_feats["max"] == {0, 1, 3, 4}
    assert best_feats["mid"] == {0, 1, 4}


def test_process_outer_loop(feature_selector, outer_loop_results):
    processed = feature_selector._process_outer_loop(outer_loop_results)
    assert processed["avg_feature_ranks"]
    assert processed["scores"]
    assert processed["n_feats"]
    assert processed["n_feats"]["min"] == 2
    assert processed["n_feats"]["max"] == 2
    assert processed["n_feats"]["mid"] == 2


def test_perform_outer_loop_cv(feature_selector):
    splits = feature_selector._make_splits()
    i = 0
    olcv = feature_selector._perform_outer_loop_cv(i, splits).compute()
    assert olcv["test_results"]
    assert olcv["scores"] is not None
    assert olcv["test_results"]["min"]
    assert olcv["test_results"]["max"]
    assert olcv["test_results"]["mid"]
    assert olcv["test_results"]["mid"]["feature_ranks"]
    assert olcv["test_results"]["mid"]["feature_ranks"]
    assert olcv["test_results"]["mid"]["score"] is not None


def test_select_best_features(feature_selector_mosquito, outer_loop_aggregation):
    selected_features = feature_selector_mosquito._select_best_features(
        outer_loop_aggregation
    )
    assert selected_features
    assert "min" in selected_features
    assert "mid" in selected_features
    assert "max" in selected_features


def test_perform_inner_loop_cv(feature_selector):
    splits = feature_selector._make_splits()
    i = 0
    res = feature_selector._perform_inner_loop_cv(i, splits)
    assert len(res) == 4
    lengths = tuple(sorted(len(feats) for feats in res))
    assert lengths == (2, 3, 4, 5)
    inner_results = list(res.values())[0]
    assert isinstance(inner_results, list)
    assert len(inner_results) == feature_selector.n_inner
    assert "score" in inner_results[0]
    assert "feature_ranks" in inner_results[0]


def test_train_and_evaluate_on_segments(feature_selector):
    split = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13], [3, 4, 5]
    features = [0, 4]
    res = feature_selector._train_and_evaluate_on_segments(split, features)
    assert res["score"] is not None
    assert res["feature_ranks"]
    assert res["feature_ranks"][0] < res["feature_ranks"][4]


def test_keep_best_features(feature_selector_mosquito, inner_results):
    features = [0, 1, 2, 3, 4, 5]
    kept = feature_selector_mosquito._keep_best_features(inner_results, features)
    kept = tuple(sorted(kept))
    assert kept == (0, 1, 2, 3, 4)


def test_extract_feature_rank(feature_selector_mosquito):
    class DummyEstimatorFI:
        feature_importances_ = None

    class DummyEstimatorCoeff:
        coef_ = None

    estimator = DummyEstimatorFI()
    feature_importances = np.array([0.05, 0.4, 0, 0, 0.15, 0.4])
    estimator.feature_importances_ = feature_importances
    features = [7, 8, 9, 10, 11, 12]
    ranks = feature_selector_mosquito._extract_feature_rank(estimator, features)

    assert len(ranks) == len(estimator.feature_importances_)
    assert ranks[7] == 4
    assert ranks[8] == 1.5
    assert ranks[9] == 5.5
    assert ranks[10] == 5.5
    assert ranks[11] == 3
    assert ranks[12] == 1.5

    estimator = DummyEstimatorCoeff()
    coef = np.array([-0.05, 0.4, 0, 0, 0.15, -0.4])
    estimator.coef_ = [coef]

    ranks = feature_selector_mosquito._extract_feature_rank(estimator, features)

    assert len(ranks) == len(coef)
    assert ranks[7] == 4
    assert ranks[8] == 1.5
    assert ranks[9] == 5.5
    assert ranks[10] == 5.5
    assert ranks[11] == 3
    assert ranks[12] == 1.5


def _select_best_features_and_score(feature_selector, outer_train_results):
    res = feature_selector._select_best_features_and_score(outer_train_results)
    for key in ("min", "max", "mid", "score"):
        assert key in res
    assert sorted(res["min"]) == (0, 1)
    assert len(res["score"]) == 4
    assert res["score"][2] == 3
    assert res["score"][5] == 3


def test_compute_avg_feature_rank(feature_selector_mosquito, outer_loop_results):
    avg_rks = feature_selector_mosquito._compute_avg_feature_rank(outer_loop_results)
    assert avg_rks["min"]
    assert avg_rks["max"]
    assert avg_rks["mid"]
    avg_rks = avg_rks["min"]
    assert avg_rks[0] == 2.5
    assert avg_rks[1] == 2.5
    assert avg_rks[2] == 2.5
    assert avg_rks[3] == 4.5
    assert avg_rks[4] == 5


def test_compute_number_of_features(feature_selector_mosquito, scores):
    n_feats = feature_selector_mosquito._compute_number_of_features(scores)
    assert isinstance(n_feats, dict)
    assert feature_selector_mosquito.MIN in n_feats
    assert feature_selector_mosquito.MAX in n_feats
    assert feature_selector_mosquito.MID in n_feats
    assert n_feats[feature_selector_mosquito.MIN] == 4
    assert n_feats[feature_selector_mosquito.MAX] == 4
    assert n_feats[feature_selector_mosquito.MID] == 4


def test_average_scores(feature_selector_mosquito, scores):
    avg_score = feature_selector_mosquito._average_scores(scores)
    assert avg_score
    assert avg_score[5] == 7.5
    assert avg_score[4] == 6.5
    assert avg_score[3] == 7
    assert avg_score[2] == 7.5


def test_normalize_score(feature_selector_mosquito):
    avg_score = {1: 11, 2: 6, 3: 1, 4: 6, 5: 11}
    norm_score = feature_selector_mosquito._normalize_score(avg_score)
    assert norm_score
    assert norm_score[1] == 1
    assert norm_score[3] == 0
    assert norm_score[2] == 0.5


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


def test_make_estimator(feature_selector_mosquito):
    random_forest = feature_selector_mosquito._make_estimator("RFC")
    assert isinstance(random_forest, RandomForestClassifier)
    random_forest_2 = feature_selector_mosquito._make_estimator(random_forest)
    assert random_forest is random_forest_2


def test_make_splits(feature_selector_mosquito):
    splits = feature_selector_mosquito._make_splits()
    n_outer_splits = feature_selector_mosquito.n_outer
    n_inner_splits = (
        feature_selector_mosquito.n_outer * feature_selector_mosquito.n_inner
    )
    assert splits
    assert len(splits) == n_inner_splits + n_outer_splits

    # check test size and train size are as expected
    possible_test_size = {
        np.floor(
            feature_selector_mosquito.X.shape[0] / feature_selector_mosquito.n_outer
        ),
        np.ceil(
            feature_selector_mosquito.X.shape[0] / feature_selector_mosquito.n_outer
        ),
    }
    possible_train_size = {
        feature_selector_mosquito.X.shape[0] - i for i in possible_test_size
    }
    for i in range(feature_selector_mosquito.n_outer):
        train_idx, test_idx = splits[(i,)]
        assert len(train_idx) in possible_train_size
        assert len(test_idx) in possible_test_size

    # check there is no ntersection among the groups
    for i in range(feature_selector_mosquito.n_outer):
        train_idx, test_idx = splits[(i,)]
        for j in range(feature_selector_mosquito.n_inner):
            inner_train, valid_idx = splits[(i, j)]
            assert not set(inner_train).intersection(valid_idx)
            assert not set(test_idx).intersection(valid_idx)
            assert not set(inner_train).intersection(test_idx)


def test_make_metric(feature_selector_mosquito):
    assert feature_selector_mosquito.metric
    assert feature_selector_mosquito.metric is miss_score


def test_miss_score():
    y_pred = np.array([1, 0, 0, 1])
    y_true = np.array([0, 1, 1, 0])
    assert miss_score(y_pred, y_true) == -4
    y_true = np.array([1, 0, 0, 1])
    assert miss_score(y_pred, y_true) == 0
    y_true = np.array([1, 0, 1, 0])
    assert miss_score(y_pred, y_true) == -2


def test_make_metric_from_string(feature_selector_mosquito):
    for metric_id in metrics.SCORERS:
        assert feature_selector_mosquito._make_metric_from_string(metric_id)
    feature_selector_mosquito._make_metric_from_string("MISS") is miss_score
    with pytest.raises(ValueError):
        feature_selector_mosquito._make_metric_from_string("yo")


def test_plot_validation_curves(feature_selector, results):
    feature_selector._results = results
    feature_selector._selected_features = feature_selector._process_results(results)
    ax = feature_selector.plot_validation_curves()
    assert ax
