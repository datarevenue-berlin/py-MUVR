import pytest

import collections
from py_muvr.post_processor import PostProcessor
from py_muvr.feature_evaluator import FeatureEvaluationResults
from py_muvr.data_structures import FeatureRanks, ScoreCurve, OuterLoopResults
from py_muvr.models import Estimator


@pytest.fixture
def outer_loop_results():
    return [
        OuterLoopResults(
            n_features_to_score_map={5: 100, 4: 5, 3: 4, 2: 5, 1: 100},
            min_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[1, 2], ranks=[1, 2], n_feats=5),
                model="model",
            ),
            max_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(
                    features=[1, 2, 3, 4], ranks=[2, 1, 3, 4], n_feats=5
                ),
                model="model",
            ),
            mid_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[1, 2, 3], ranks=[1, 2, 3], n_feats=5),
                model="model",
            ),
        ),
        OuterLoopResults(
            n_features_to_score_map={5: 300, 4: 6, 3: 4, 2: 7, 1: 250},
            min_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[1, 2], ranks=[1.5, 1.5], n_feats=5),
                model="model",
            ),
            max_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(
                    features=[0, 1, 2, 3], ranks=[1, 2, 3, 4], n_feats=5
                ),
                model="model",
            ),
            mid_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[0, 1, 2], ranks=[3, 1, 2], n_feats=5),
                model="model",
            ),
        ),
    ]


@pytest.fixture
def outer_loop_results2():
    return [
        OuterLoopResults(
            n_features_to_score_map={5: 150, 4: 4, 3: 4, 2: 5, 1: 120},
            min_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[2, 3], ranks=[1, 2], n_feats=5),
                model="model",
            ),
            max_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(
                    features=[0, 1, 2, 3], ranks=[3, 1, 2, 4], n_feats=5
                ),
                model="model",
            ),
            mid_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[0, 1, 2], ranks=[3, 1, 2], n_feats=5),
                model="model",
            ),
        ),
        OuterLoopResults(
            n_features_to_score_map={5: 200, 4: 7, 3: 1, 2: 6, 1: 220},
            min_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[0, 1], ranks=[1, 2], n_feats=5),
                model="model",
            ),
            max_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(
                    features=[0, 1, 2, 4], ranks=[4, 3, 1, 2], n_feats=5
                ),
                model="model",
            ),
            mid_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[1, 2, 3], ranks=[3, 1, 2], n_feats=5),
                model="model",
            ),
        ),
    ]


@pytest.fixture
def repetitions(outer_loop_results, outer_loop_results2):
    return [outer_loop_results, outer_loop_results2]


def test_post_processor():
    pp = PostProcessor(robust_minimum=0.05)
    assert pp


def test_select_features(repetitions):
    pp = PostProcessor(robust_minimum=0.05)
    selected_feats = pp.select_features(repetitions)
    # the scores have a deep minimum at 2, 3 and 4 features
    assert selected_feats["min"]
    assert selected_feats["mid"]
    assert selected_feats["max"]
    assert sorted(selected_feats["min"]) == [1, 2]
    assert sorted(selected_feats["mid"]) == [1, 2, 3]
    assert sorted(selected_feats["max"]) == [0, 1, 2, 3]


def test_compute_n_features(repetitions):
    pp = PostProcessor(robust_minimum=0.05)
    n_feats = pp._compute_n_features(repetitions)
    # the scores have a deep minimum at 2, 3 and 4 features
    assert len(n_feats) == 3
    min_feats, mid_feats, max_feats = n_feats
    assert min_feats == 2
    assert mid_feats == 3
    assert max_feats == 4


def test_get_repetition_avg_scores(repetitions):
    pp = PostProcessor(robust_minimum=0.05)
    avg_scores = pp._get_repetition_avg_scores(repetitions)
    assert len(avg_scores) == 2
    assert avg_scores[0][5] == 200
    assert avg_scores[0][4] == 5.5
    assert avg_scores[0][3] == 4
    assert avg_scores[0][2] == 6
    assert avg_scores[0][1] == 175


def test_get_validation_curves(repetitions):
    pp = PostProcessor(robust_minimum=0.05)
    curves = pp.get_validation_curves(repetitions)
    assert len(curves) == 3
    assert len(curves["outer_loops"]) == 4
    assert len(curves["repetitions"]) == 2
    assert len(curves["total"]) == 1
    assert isinstance(curves["total"][0], ScoreCurve)
    assert sorted(curves["outer_loops"][0].n_features) == [1, 2, 3, 4, 5]
    assert sorted(curves["repetitions"][0].n_features) == [1, 2, 3, 4, 5]
    assert list(curves["repetitions"][0].scores) == [175, 6, 4, 5.5, 200]


def test_process_feature_elim_results(rfe_raw_results):
    pp = PostProcessor(0.05)

    processed = pp.process_feature_elim_results(rfe_raw_results)

    assert processed.best_features
    assert processed.n_features_to_score_map


def test_compute_score_curve(rfe_raw_results):
    pp = PostProcessor(0.05)

    avg_scores = pp._compute_score_curve(rfe_raw_results)

    assert len(avg_scores) == 3
    assert 2 in avg_scores
    assert 3 in avg_scores
    assert 4 in avg_scores
    assert avg_scores[4] < avg_scores[3]


def test_select_best_features(rfe_raw_results):
    pp = PostProcessor(1)
    avg_scores = pp._compute_score_curve(rfe_raw_results)

    selected_feats = pp._select_best_outer_features(rfe_raw_results, avg_scores)

    assert sorted(selected_feats["min"]) == [2, 4]
    assert sorted(selected_feats["mid"]) == [2, 3, 4]
    assert sorted(selected_feats["max"]) == [1, 2, 3, 4]


def test_make_average_ranks_dataframe(fs_results):
    pp = PostProcessor(1)
    n_feats = 5
    feature_names = "a, b, c, d, e".split(", ")

    ranks_df = pp.make_average_ranks_df(fs_results, n_feats, feature_names)

    assert ranks_df.ndim
    assert len(ranks_df) == n_feats
    assert set(ranks_df.index) == set(feature_names)


def test_exclude_unused_features(fs_results):
    pp = PostProcessor(1)
    n_feats = 5
    unused_feats = 10

    reduced_df = pp.make_average_ranks_df(
        fs_results, unused_feats, exclude_unused_features=True
    )
    full_df = pp.make_average_ranks_df(
        fs_results, unused_feats, exclude_unused_features=False
    )

    assert reduced_df.ndim
    assert full_df.ndim
    assert len(reduced_df) == n_feats
    assert len(full_df) == unused_feats


def test_get_feature_ranks(raw_results):
    pp = PostProcessor(1)

    min_ranks = pp._get_feature_ranks(raw_results, "min")
    max_ranks = pp._get_feature_ranks(raw_results, "max")

    assert len(min_ranks) == len(max_ranks)
    assert isinstance(min_ranks[0], dict)
