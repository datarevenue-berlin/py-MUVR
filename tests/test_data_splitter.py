import pytest
import numpy as np
from omigami.models import InputData
from omigami.data_splitter import DataSplitter


@pytest.fixture
def dataset():
    X = np.zeros((10, 10))
    y = np.zeros(10)
    return InputData(X=X, y=y, groups=np.arange(10))


@pytest.fixture
def grouped_dataset():
    X = np.zeros((10, 10))
    y = np.arange(10)
    groups = y // 2
    return InputData(X=X, y=y, groups=groups)


def test_data_splitter(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, input_data=dataset, random_state=0)
    assert ds


def test_make_splits(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, input_data=dataset, random_state=0)
    assert ds._splits
    assert len(ds._splits) == 5 * 4 + 5
    assert len(ds._splits[(0, None)]) == 2
    assert len(ds._splits[(0, 2)]) == 2
    train_idx, test_idx = ds._splits[(0, 2)]
    assert len(train_idx) == 6
    assert len(test_idx) == 2
    train_idx, test_idx = ds._splits[(0, None)]
    assert len(train_idx) == 8
    assert len(test_idx) == 2
    assert not set(train_idx).intersection(test_idx)


def test_make_splits_grouped(grouped_dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, input_data=grouped_dataset, random_state=0)
    assert ds


def test_get_outer_splits(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, input_data=dataset, random_state=0)
    outer_split = ds.get_outer_split(0)
    assert len(outer_split) == 2
    assert outer_split == ds._splits[(0, None)]


def test_get_inner_splits(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, input_data=dataset, random_state=0)
    inner_split = ds.get_inner_split(0, 0)
    assert len(inner_split) == 2
    assert inner_split == ds._splits[(0, 0)]
