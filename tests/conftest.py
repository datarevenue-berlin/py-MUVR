import pytest
from omigami.outer_looper import OuterLoopResults, OuterLoopModelTrainResults
from omigami.model_trainer import TrainingTestingResult


@pytest.fixture(scope="session")
def results():
    return [
        [
            OuterLoopResults(
                test_results=OuterLoopModelTrainResults(
                    MIN=TrainingTestingResult(score=4, feature_ranks={0: 1.0, 1: 2.0}),
                    MAX=TrainingTestingResult(
                        score=5, feature_ranks={0: 1.0, 1: 2.0, 2: 4.0, 3: 3.0}
                    ),
                    MID=TrainingTestingResult(
                        score=5, feature_ranks={0: 1.0, 1: 2.0, 3: 3.0}
                    ),
                ),
                scores={5: 4, 4: 3, 3: 3, 2: 3},
            ),
            OuterLoopResults(
                test_results=OuterLoopModelTrainResults(
                    MIN=TrainingTestingResult(
                        score=3, feature_ranks={0: 1.0, 1: 2.0, 4: 3.0, 3: 4.0}
                    ),
                    MAX=TrainingTestingResult(
                        score=3, feature_ranks={0: 1.0, 1: 2.0, 4: 3.0, 3: 4.0}
                    ),
                    MID=TrainingTestingResult(
                        score=2, feature_ranks={0: 1.0, 1: 2.0, 4: 3.0, 3: 4.0}
                    ),
                ),
                scores={5: 5, 4: 4, 3: 5, 2: 5},
            ),
        ],
        [
            OuterLoopResults(
                test_results=OuterLoopModelTrainResults(
                    MIN=TrainingTestingResult(score=4, feature_ranks={0: 1.0, 1: 2.0}),
                    MAX=TrainingTestingResult(
                        score=5, feature_ranks={0: 1.0, 1: 2.0, 4: 3.0, 2: 4.0}
                    ),
                    MID=TrainingTestingResult(
                        score=5, feature_ranks={0: 2.0, 1: 1.0, 4: 3.0}
                    ),
                ),
                scores={5: 5, 4: 3, 3: 5, 2: 3},
            ),
            OuterLoopResults(
                test_results=OuterLoopModelTrainResults(
                    MIN=TrainingTestingResult(score=2, feature_ranks={0: 1.0, 1: 2.0}),
                    MAX=TrainingTestingResult(
                        score=2, feature_ranks={0: 1.0, 1: 2.0, 2: 5.0, 3: 4.0, 4: 3.0}
                    ),
                    MID=TrainingTestingResult(
                        score=2, feature_ranks={0: 1.0, 1: 2.0, 4: 3.0}
                    ),
                ),
                scores={5: 5, 4: 6, 3: 5, 2: 5},
            ),
        ],
    ]

