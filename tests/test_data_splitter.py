import pytest
import numpy as np
from py_muvr.data_structures import InputDataset, Split
from py_muvr.data_splitter import DataSplitter


@pytest.fixture
def grouped_dataset():
    X = np.zeros((10, 10))
    y = np.arange(10)
    groups = y // 2
    return InputDataset(X=X, y=y, groups=groups)


def test_data_splitter(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=dataset)
    assert ds
    assert ds._splits


def test_make_splits(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=dataset)
    assert ds._splits
    assert len(ds._splits) == 5 * 4 + 5
    split = ds._splits[(0, 2)]
    train_idx = split.train_indices
    test_idx = split.test_indices
    assert len(train_idx) == 6
    assert len(test_idx) == 3
    split = ds._splits[(0, None)]
    train_idx = split.train_indices
    test_idx = split.test_indices
    assert len(train_idx) == 9
    assert len(test_idx) == 3
    assert not set(train_idx).intersection(test_idx)


def test_split_separation(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=dataset)
    # all indices should appear once if joining outer_test, inner_test and inner_train
    for outer_split in ds.iter_outer_splits():
        for inner_split in ds.iter_inner_splits(outer_split):
            all_indeces = (
                list(outer_split.test_indices)
                + list(inner_split.test_indices)
                + list(inner_split.train_indices)
            )
            assert len(all_indeces) == 12
            assert sorted(all_indeces) == list(range(12))
            out_train = list(inner_split.test_indices) + list(inner_split.train_indices)
            assert sorted(out_train) == sorted(outer_split.train_indices)


def test_make_splits_grouped(grouped_dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=grouped_dataset)
    groups = grouped_dataset.groups
    assert ds
    # check there is no intersection among the groups
    for outer_split in ds.iter_outer_splits():
        train_idx = outer_split.train_indices
        test_idx = outer_split.test_indices
        for inner_split in ds.iter_inner_splits(outer_split):
            inner_train = inner_split.train_indices
            valid_idx = inner_split.test_indices
            assert not set(groups[inner_train]).intersection(groups[valid_idx])
            assert not set(groups[test_idx]).intersection(groups[valid_idx])
            assert not set(groups[inner_train]).intersection(groups[test_idx])


def test_iter_outer_splits(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=dataset)
    for outer_split in ds.iter_outer_splits():
        assert outer_split
        assert outer_split == ds._splits[(outer_split.id, None)]


def test_iter_inner_splits(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=dataset)
    for outer_split in ds.iter_outer_splits():
        for inner_split in ds.iter_inner_splits(outer_split):
            assert inner_split
            assert inner_split == ds._splits[(outer_split.id, inner_split.id)]


def test_randomness(dataset):
    random_state = np.random.RandomState(0)
    ds = DataSplitter(
        n_outer=5, n_inner=4, random_state=random_state, input_data=dataset
    )
    ds2 = DataSplitter(
        n_outer=5, n_inner=4, random_state=random_state, input_data=dataset
    )
    assert all(
        sorted(ds._splits[k].train_indices) != sorted(ds2._splits[k].train_indices)
        for k in ds._splits
    )


def test_non_randomness(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=dataset)
    ds2 = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=dataset)
    assert all(
        sorted(ds._splits[k].train_indices) == sorted(ds2._splits[k].train_indices)
        for k in ds._splits
    )


def test_split_data(dataset):
    ds = DataSplitter(n_outer=5, n_inner=4, random_state=0, input_data=dataset)
    split = Split(0, [0, 1, 2], [3, 4, 5, 6])
    split_data = ds.split_data(dataset, split, features=[0, 1, 2])
    assert split_data.test_data.X.shape == (4, 3)
    assert split_data.test_data.y.size == 4
    assert split_data.train_data.X.shape == (3, 3)
    assert split_data.train_data.y.size == 3
