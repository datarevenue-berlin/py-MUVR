import pytest
import numpy as np
from omigami.outer_loop import OuterLoop, OuterLoopResults
from omigami.feature_evaluator import FeatureEvaluator, FeatureEvaluationResults
from omigami.data_models import InputData


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


def test_run(feature_evaluator):
    out_loop = OuterLoop(4, feature_evaluator, 0.5, 0.001)
    res = out_loop.run()
    assert len(res) == 4
    assert all(isinstance(r, OuterLoopResults) for r in res)


def testexecute_loop(feature_evaluator):
    out_loop = OuterLoop(4, feature_evaluator, 0.5, 0.001)
    results = out_loop.execute_loop(0)
    assert results
    assert isinstance(results, OuterLoopResults)
    assert len(results.score_vs_feats) == 4
