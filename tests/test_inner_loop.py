import pytest
import numpy as np
from omigami.inner_loop import InnerLoop
from omigami.models import InputData
from omigami.feature_evaluator import FeatureEvaluator, FeatureEvaluationResults


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
    in_loop = InnerLoop(feature_evaluator)
    res = in_loop.run([1, 2, 3], 0)
    assert res
    assert len(res) == 3
    assert isinstance(res[0], FeatureEvaluationResults)
    assert res[0].ranks[1] < 12
    assert res[0].ranks[2] < 12
    assert res[0].ranks[3] < 12
    assert res[0].ranks[10] == 12




