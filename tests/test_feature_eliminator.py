import pytest
import numpy as np
from omigami.models import InputData, FeatureRanks, FeatureEvaluationResults
from omigami.feature_evaluator import FeatureEvaluator
from omigami.recursive_feature_eliminator import RecursiveFeatureEliminator, RecursiveFeatureEliminationResults



@pytest.fixture
def dataset():
    X = np.random.rand(12, 12)
    y = np.random.choice([0, 1], 12)
    return InputData(X=X, y=y, groups=np.arange(12))


@pytest.fixture
def feature_evaluator(dataset):
    fe = FeatureEvaluator(
        n_inner=3,
        n_outer=4,
        estimator="RFC",
        metric="MISS",
        input_data=dataset,
        random_state=0,
    )
    return fe


@pytest.fixture
def inner_loop_results():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3], ranks=[3, 2, 1]),
            test_score=0.2,
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3], ranks=[1.5, 1.5, 3]),
            test_score=0.2,
        ),
    ]


@pytest.fixture
def inner_loop_results():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3], ranks=[3, 2, 1]),
            test_score=0.2,
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3], ranks=[1.5, 1.5, 3]),
            test_score=0.2,
        ),
    ]

@pytest.fixture
def inner_loop_results_2():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1], ranks=[1]),
            test_score=0.4,
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1], ranks=[1]),
            test_score=0.4,
        ),
    ]


def test_feature_eliminator(feature_evaluator):
    rfe = RecursiveFeatureEliminator(feature_evaluator, dropout_rate=0.1, robust_minimum=0.05)
    assert rfe


def test_run(feature_evaluator):
    rfe = RecursiveFeatureEliminator(feature_evaluator, dropout_rate=0.5, robust_minimum=0.05)
    # 12 input features halved at every round 12 -> 6 -> 3 -> 1
    results = rfe.run(outer_loop_index=0)
    assert results
    assert isinstance(results, RecursiveFeatureEliminationResults)
    assert results.best_feats
    assert results.score_vs_feats
    assert len(results.score_vs_feats) == 4
    assert len(results.score_vs_feats[(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)]) == feature_evaluator.get_inner_loop_size()


@pytest.mark.parametrize(
    "keep, selected", [(2, [2, 3]), (1, [2])]
)
def test_remove_features(feature_evaluator, inner_loop_results, keep, selected):
    rfe = RecursiveFeatureEliminator(feature_evaluator, dropout_rate=0.3, robust_minimum=0.05)
    selected_features = rfe._remove_features(inner_loop_results, keep)
    assert len(selected_features) == keep
    assert sorted(selected_features) == selected  # lowest avg ranks


def test_compute_score_curve(inner_loop_results, inner_loop_results_2, feature_evaluator):
    rfe_res = {(1, 2, 3): inner_loop_results, (1,): inner_loop_results_2}
    rfe = RecursiveFeatureEliminator(feature_evaluator, dropout_rate=0.3, robust_minimum=0.05)
    avg_scores = rfe._compute_score_curve(rfe_res)
    assert len(avg_scores) == 2
    assert 1 in avg_scores
    assert 3 in avg_scores
    assert avg_scores[3] < avg_scores[1]


def test_select_best_features(inner_loop_results, inner_loop_results_2, feature_evaluator):
    rfe_res = {(1, 2, 3): inner_loop_results, (1,): inner_loop_results_2}
    rfe = RecursiveFeatureEliminator(feature_evaluator, dropout_rate=0.3, robust_minimum=0.05)
    selected_feats = rfe._select_best_features(rfe_res)
    # edge case: it's just two recursive steps at 3 and 1 features. The one at
    # 3 is the best (lowest test-score) so every feature set should be [1, 2, 3]
    assert sorted(selected_feats.min_feats) == [1, 2, 3]
    assert sorted(selected_feats.mid_feats) == [1, 2, 3]
    assert sorted(selected_feats.max_feats) == [1, 2, 3]
