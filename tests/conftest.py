from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_muvr import FeatureSelector
from py_muvr.data_structures import (
    FeatureEvaluationResults,
    FeatureRanks,
    InputDataset,
    OuterLoopResults,
)

ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def raw_results():
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


@pytest.fixture(scope="session")
def mosquito():
    df = pd.read_csv(ASSETS_DIR / "mosquito.csv", index_col=0)
    df = df.sample(frac=1)
    X = df.drop(columns=["Yotu"]).values
    y = df.Yotu.values
    groups = df.index
    return InputDataset(X=X, y=y, groups=groups)


@pytest.fixture(scope="session")
def freelive():
    df = pd.read_csv(ASSETS_DIR / "freelive.csv", index_col=0)
    X = df.drop(columns=["YR"]).values
    y = df.YR.values
    groups = df.index
    return InputDataset(X=X, y=y, groups=groups)


@pytest.fixture(scope="session")
def fs_results(raw_results):
    fs = FeatureSelector(n_outer=3, metric="MISS", estimator="RFC")
    fs._raw_results = raw_results
    fs.is_fit = True
    fs._selected_features = fs._post_processor.select_features(raw_results)
    fs._n_features = 5
    fs_results = fs.get_feature_selection_results(["A", "B", "C", "D", "E"])
    return fs_results
