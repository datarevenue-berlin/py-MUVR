import numpy as np
import pytest

from data_models import FeatureEvaluationResults, FeatureRanks, InputData
from omigami.outer_looper import OuterLoopResults, OuterLoopModelTrainResults
from omigami.model_trainer import TrainingTestingResult, FeatureRanks


@pytest.fixture(scope="session")
def results():
    return [
        [
            OuterLoopResults(
                test_results=OuterLoopModelTrainResults(
                    MIN=TrainingTestingResult(
                        score=4,
                        feature_ranks=FeatureRanks(features=[0, 1], ranks=[1, 2]),
                    ),
                    MAX=TrainingTestingResult(
                        score=5,
                        feature_ranks=FeatureRanks(
                            features=[0, 1, 2, 3], ranks=[1, 2, 4, 3]
                        ),
                    ),
                    MID=TrainingTestingResult(
                        score=5,
                        feature_ranks=FeatureRanks(features=[0, 1, 3], ranks=[1, 2, 3]),
                    ),
                ),
                scores={5: 4, 4: 3, 3: 3, 2: 3},
            ),
            OuterLoopResults(
                test_results=OuterLoopModelTrainResults(
                    MIN=TrainingTestingResult(
                        score=3,
                        feature_ranks=FeatureRanks(
                            features=[0, 1, 4, 3], ranks=[1, 2, 3, 4]
                        ),
                    ),
                    MAX=TrainingTestingResult(
                        score=3,
                        feature_ranks=FeatureRanks(
                            features=[0, 1, 4, 3], ranks=[1, 2, 3, 4]
                        ),
                    ),
                    MID=TrainingTestingResult(
                        score=2,
                        feature_ranks=FeatureRanks(
                            features=[0, 1, 4, 3], ranks=[1, 2, 3, 4]
                        ),
                    ),
                ),
                scores={5: 5, 4: 4, 3: 5, 2: 5},
            ),
        ],
        [
            OuterLoopResults(
                test_results=OuterLoopModelTrainResults(
                    MIN=TrainingTestingResult(
                        score=4,
                        feature_ranks=FeatureRanks(features=[0, 1], ranks=[1, 2]),
                    ),
                    MAX=TrainingTestingResult(
                        score=5,
                        feature_ranks=FeatureRanks(
                            features=[0, 1, 4, 2], ranks=[1, 2, 3, 4]
                        ),
                    ),
                    MID=TrainingTestingResult(
                        score=5,
                        feature_ranks=FeatureRanks(features=[0, 1, 4], ranks=[2, 1, 3]),
                    ),
                ),
                scores={5: 5, 4: 3, 3: 5, 2: 3},
            ),
            OuterLoopResults(
                test_results=OuterLoopModelTrainResults(
                    MIN=TrainingTestingResult(
                        score=2,
                        feature_ranks=FeatureRanks(features=[0, 1], ranks=[1, 2]),
                    ),
                    MAX=TrainingTestingResult(
                        score=2,
                        feature_ranks=FeatureRanks(
                            features=[0, 1, 2, 3, 4], ranks=[1, 2, 5, 4, 3]
                        ),
                    ),
                    MID=TrainingTestingResult(
                        score=2,
                        feature_ranks=FeatureRanks(features=[0, 1, 4], ranks=[1, 2, 3]),
                    ),
                ),
                scores={5: 5, 4: 6, 3: 5, 2: 5},
            ),
        ],
    ]


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


@pytest.fixture
def inner_loop_results_2():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[2, 3, 4], ranks=[3, 2, 1]), test_score=0.1,
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[2, 3, 4], ranks=[1.5, 1.5, 3]), test_score=0.5,
        ),
    ]


@pytest.fixture
def inner_loop_results_3():
    return [
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[2, 4], ranks=[3, 2, 1]), test_score=0.3,
        ),
        FeatureEvaluationResults(
            ranks=FeatureRanks(features=[2, 4], ranks=[1.5, 1.5, 3]), test_score=0.25,
        ),
    ]


@pytest.fixture
def rfe_raw_results(inner_loop_results, inner_loop_results_2, inner_loop_results_3):
    return {
        (1, 2, 3, 4): inner_loop_results,
        (2, 3, 4): inner_loop_results_2,
        (2, 4): inner_loop_results_3
    }


@pytest.fixture
def dataset():
    X = np.random.rand(12, 12)
    y = np.random.choice([0, 1], 12)
    return InputData(X=X, y=y, groups=np.arange(12))
