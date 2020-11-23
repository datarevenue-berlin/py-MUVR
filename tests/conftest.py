import pytest
from omigami.outer_looper import OuterLoopResults, OuterLoopModelTrainResults
from omigami.model_trainer import TrainingTestingResult, FeatureRanks
from omigami.omigami import FeatureSelector


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


@pytest.fixture(scope="session")
def fitted_feature_selector(results):
    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="RFC",)
    fs.n_features = 10
    fs.is_fit = True
    fs.selected_features = {"min": (0, 1), "max": (0, 1), "mid": (0, 1)}
    fs._results = results
    sel_feats = fs._process_results(results)
    fs._selected_features = sel_feats
    return fs
