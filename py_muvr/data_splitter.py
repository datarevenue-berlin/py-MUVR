from typing import Union, Dict, List, Iterable

from py_muvr.data_structures import (
    RandomState,
    InputDataset,
    Split,
    TrainTestData,
)
from sklearn.model_selection import GroupShuffleSplit


class DataSplitter:
    """
    Class used to create the inner and outer splits for the double cross-validation and
    to slice the data using those splits and subsets of the features.

    Parameters
    ----------
    n_outer: int
        Number of outer splits to create
    n_inner:
        Number of inner splits to create
    input_data: InputDataset
        Object containing X, Y and groups attributes
    random_state: RandomState
        a random state instance to control reproducibility

    """

    def __init__(
        self,
        n_outer: int,
        n_inner: int,
        input_data: InputDataset,
        random_state: Union[int, RandomState],
    ):
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.random_state = random_state
        self._splits = self._make_splits(input_data)

    def _make_splits(self, input_data: InputDataset) -> Dict[tuple, Split]:
        """Create inner and outer splits.

        Creates nested splits of the input data. The outer training datasets
        will be split again to provide validation and inner training datasets.
        """
        outer_splitter = self._make_random_splitter(self.n_outer)
        inner_splitter = self._make_random_splitter(self.n_inner)

        outer_splits = outer_splitter.split(
            input_data.X, input_data.y, input_data.groups
        )

        splits = {}
        for out_idx, (out_train, out_test) in enumerate(outer_splits):
            splits[(out_idx, None)] = Split(out_idx, out_train, out_test)
            X_train = input_data.X[out_train, :]
            y_train = input_data.y[out_train]
            groups_train = input_data.groups[out_train]
            inner_splits = inner_splitter.split(X_train, y_train, groups_train)
            for in_idx, (in_train, in_test) in enumerate(inner_splits):
                inner_split = Split(in_idx, out_train[in_train], out_train[in_test])
                splits[(out_idx, in_idx)] = inner_split
        return splits

    def _make_random_splitter(self, n_splits):
        test_size = 1 / n_splits
        return GroupShuffleSplit(
            n_splits, test_size=test_size, random_state=self.random_state
        )

    def iter_outer_splits(self) -> Iterable[Split]:
        """
        Iterates through the splits corresponding to the outer loops

        Returns
        -------
        Outer loop split used to slice the input dataset
        """
        for outer_idx in range(self.n_outer):
            yield self._splits[(outer_idx, None)]

    def iter_inner_splits(self, outer_split: Split) -> Iterable[Split]:
        """
        Given an outer split, iterates through the splits corresponding to the outer
        split's inner loops.

        Parameters
        ----------
        outer_split: Split
            Split from calling the iter_outer_splits method.

        Returns
        -------
        Split:
            Inner loop split used to slice the input dataset

        """
        outer_idx = outer_split.id
        for inner_idx in range(self.n_inner):
            yield self._splits[(outer_idx, inner_idx)]

    @staticmethod
    def split_data(
        input_data: InputDataset, split: Split, features: List[int] = None
    ) -> TrainTestData:
        """
        Splits the input dataset into train and test sets, optionally selecting a subset
        of features.

        Parameters
        ----------
        input_data: InputDataset
            Object containing X, Y and groups attributes
        split: Split
            Split from calling the iter splits methods
        features: List[int]
            List of features to select from the input data

        Returns
        -------
        TrainTestData:
            The sliced input data.
        """
        return TrainTestData(
            train_data=input_data[split.train_indices, features],
            test_data=input_data[split.test_indices, features],
        )
