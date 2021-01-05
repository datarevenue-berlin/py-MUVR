import pytest
import numpy as np
from omigami.models import InputData


@pytest.fixture
def dataset():
    X = np.zeros((10, 10))
    y = np.zeros(10)
    return InputData(X=X, y=y, groups=np.arange(10))


def test_input_data_slice(dataset):
    X, y = dataset.slice_data(indices=[0, 1, 2], features=[0, 1, 2])
    assert np.all(X == dataset.X[[0, 1, 2], :][:, [0, 1, 2]])
    assert np.all(y == dataset.y[[0, 1, 2]])
