import pytest
from omigami.post_processor import PostProcessor
from omigami.outer_loop import OuterLoopResults
from omigami.feature_evaluator import FeatureEvaluationResults
from omigami.models import FeatureRanks, ScoreCurve


@pytest.fixture
def outer_loop_results():
    return [
        OuterLoopResults(
            score_vs_feats={5: 100, 4: 5, 3: 4, 2: 5, 1: 100},
            min_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[1, 2], ranks=[1, 2], n_feats=5),
            ),
            max_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(
                    features=[1, 2, 3, 4], ranks=[2, 1, 3, 4], n_feats=5
                ),
            ),
            mid_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[1, 2, 3], ranks=[1, 2, 3], n_feats=5),
            ),
        ),
        OuterLoopResults(
            score_vs_feats={5: 300, 4: 6, 3: 4, 2: 7, 1: 250},
            min_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[1, 2], ranks=[1.5, 1.5], n_feats=5),
            ),
            max_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(
                    features=[0, 1, 2, 3], ranks=[1, 2, 3, 4], n_feats=5
                ),
            ),
            mid_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[0, 1, 2], ranks=[3, 1, 2], n_feats=5),
            ),
        ),
    ]


@pytest.fixture
def outer_loop_results2():
    return [
        OuterLoopResults(
            score_vs_feats={5: 150, 4: 4, 3: 4, 2: 5, 1: 120},
            min_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[2, 3], ranks=[1, 2], n_feats=5),
            ),
            max_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(
                    features=[0, 1, 2, 3], ranks=[3, 1, 2, 4], n_feats=5
                ),
            ),
            mid_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[0, 1, 2], ranks=[3, 1, 2], n_feats=5),
            ),
        ),
        OuterLoopResults(
            score_vs_feats={5: 200, 4: 7, 3: 1, 2: 6, 1: 220},
            min_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[0, 1], ranks=[1, 2], n_feats=5),
            ),
            max_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(
                    features=[0, 1, 2, 4], ranks=[4, 3, 1, 2], n_feats=5
                ),
            ),
            mid_eval=FeatureEvaluationResults(
                test_score=0,
                ranks=FeatureRanks(features=[1, 2, 3], ranks=[3, 1, 2], n_feats=5),
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
    assert selected_feats.min_feats
    assert selected_feats.mid_feats
    assert selected_feats.max_feats
    assert sorted(selected_feats.min_feats) == [1, 2]
    assert sorted(selected_feats.mid_feats) == [1, 2, 3]
    assert sorted(selected_feats.max_feats) == [0, 1, 2, 3]


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
