import numpy as np
import pytest

from omigami.data import (
    FeatureEvaluationResults,
    FeatureRanks,
    InputDataset,
    OuterLoopResults,
)


@pytest.fixture(scope="session")
def results():
    return [
        [
            OuterLoopResults(
                min_eval=FeatureEvaluationResults(
                    test_score=4,
                    model="model",
                    ranks=FeatureRanks(features=[0, 1], ranks=[1, 2], n_feats=10),
                ),
                max_eval=FeatureEvaluationResults(
                    test_score=5,
                    model="model",
                    ranks=FeatureRanks(
                        features=[0, 1, 2, 3], ranks=[1, 2, 4, 3], n_feats=10
                    ),
                ),
                mid_eval=FeatureEvaluationResults(
                    test_score=5,
                    model="model",
                    ranks=FeatureRanks(features=[0, 1, 3], ranks=[1, 2, 3], n_feats=10),
                ),
                n_features_to_score_map={5: 4, 4: 3, 3: 3, 2: 3},
            ),
            OuterLoopResults(
                min_eval=FeatureEvaluationResults(
                    test_score=3,
                    model="model",
                    ranks=FeatureRanks(
                        features=[0, 1, 4, 3], ranks=[1, 2, 3, 4], n_feats=10
                    ),
                ),
                max_eval=FeatureEvaluationResults(
                    test_score=3,
                    model="model",
                    ranks=FeatureRanks(
                        features=[0, 1, 4, 3], ranks=[1, 2, 3, 4], n_feats=10
                    ),
                ),
                mid_eval=FeatureEvaluationResults(
                    test_score=2,
                    model="model",
                    ranks=FeatureRanks(
                        features=[0, 1, 4, 3], ranks=[1, 2, 3, 4], n_feats=10
                    ),
                ),
                n_features_to_score_map={5: 5, 4: 4, 3: 5, 2: 5},
            ),
        ],
        [
            OuterLoopResults(
                min_eval=FeatureEvaluationResults(
                    test_score=4,
                    model="model",
                    ranks=FeatureRanks(features=[0, 1], ranks=[1, 2], n_feats=10),
                ),
                max_eval=FeatureEvaluationResults(
                    test_score=5,
                    model="model",
                    ranks=FeatureRanks(
                        features=[0, 1, 4, 2], ranks=[1, 2, 3, 4], n_feats=10
                    ),
                ),
                mid_eval=FeatureEvaluationResults(
                    test_score=5,
                    model="model",
                    ranks=FeatureRanks(features=[0, 1, 4], ranks=[2, 1, 3], n_feats=10),
                ),
                n_features_to_score_map={5: 5, 4: 3, 3: 5, 2: 3},
            ),
            OuterLoopResults(
                min_eval=FeatureEvaluationResults(
                    test_score=2,
                    model="model",
                    ranks=FeatureRanks(features=[0, 1], ranks=[1, 2], n_feats=10),
                ),
                max_eval=FeatureEvaluationResults(
                    test_score=2,
                    model="model",
                    ranks=FeatureRanks(
                        features=[0, 1, 2, 3, 4], ranks=[1, 2, 5, 4, 3], n_feats=10
                    ),
                ),
                mid_eval=FeatureEvaluationResults(
                    test_score=2,
                    model="model",
                    ranks=FeatureRanks(features=[0, 1, 4], ranks=[1, 2, 3], n_feats=10),
                ),
                n_features_to_score_map={5: 5, 4: 6, 3: 5, 2: 5},
            ),
        ],
    ]


@pytest.fixture
def inner_loop_results():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3, 4], ranks=[3, 2, 1, 4]),
            test_score=0.2,
            model="estimator",
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[1, 2, 3, 4], ranks=[1.5, 1.5, 3, 4]),
            test_score=0.2,
            model="estimator",
        ),
    ]


@pytest.fixture
def inner_loop_results_2():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[2, 3, 4], ranks=[3, 2, 1]),
            test_score=0.1,
            model="model",
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[2, 3, 4], ranks=[1.5, 1.5, 3]),
            test_score=0.5,
            model="model",
        ),
    ]


@pytest.fixture
def inner_loop_results_3():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[2, 4], ranks=[3, 2, 1]),
            test_score=0.3,
            model="model",
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[2, 4], ranks=[1.5, 1.5, 3]),
            test_score=0.25,
            model="model",
        ),
    ]


@pytest.fixture
def rfe_raw_results(inner_loop_results, inner_loop_results_2, inner_loop_results_3):
    return {
        (1, 2, 3, 4): inner_loop_results,
        (2, 3, 4): inner_loop_results_2,
        (2, 4): inner_loop_results_3,
    }


@pytest.fixture
def dataset():
    X = np.random.rand(12, 12)
    y = np.random.choice([0, 1], 12)
    return InputDataset(X=X, y=y, groups=np.arange(12))
