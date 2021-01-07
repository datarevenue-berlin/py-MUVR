import pytest
import numpy as np
from omigami.data import InputDataset, Split


@pytest.fixture
def dataset():
    X = np.zeros((10, 12))
    y = np.zeros(10)
    return InputDataset(X=X, y=y, groups=np.arange(10))


def test_input_data_slice(dataset):
    sliced = dataset._slice_data(indices=[0, 1, 2], features=[0, 1, 2])
    X = sliced.X
    y = sliced.y
    assert np.all(X == dataset.X[[0, 1, 2], :][:, [0, 1, 2]])
    assert np.all(y == dataset.y[[0, 1, 2]])


def test_split_data(dataset):
    split = Split(0, [0, 1, 2], [3, 4, 5, 6])
    split_data = dataset.split_data(split, features=[0, 1, 2])
    assert split_data.test_data.X.shape == (4, 3)
    assert split_data.test_data.y.size == 4
    assert split_data.train_data.X.shape == (3, 3)
    assert split_data.train_data.y.size == 3


def test_n_features(dataset):
    assert dataset.n_features == 12
