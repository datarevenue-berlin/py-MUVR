# TODO: rename as data_models

from typing import Iterable, List
from dataclasses import dataclass
from omigami.types import NumpyArray


@dataclass
class Split:
    id: int
    train_indices: NumpyArray
    test_indices: NumpyArray


@dataclass
class InputData:
    X: NumpyArray
    y: NumpyArray
    groups: NumpyArray

    def split_data(self, split, features=None):
        return TrainTestData(
            train_data=self._slice_data(indices=split.train_indices, features=features),
            test_data=self._slice_data(indices=split.test_indices, features=features),
        )

    def _slice_data(self, indices=None, features=None):
        X_sliced = self.X
        y_sliced = self.y
        g_sliced = self.groups
        if indices is not None:
            X_sliced = X_sliced[indices, :]
            y_sliced = y_sliced[indices]
            g_sliced = g_sliced[indices]
        if features is not None:
            X_sliced = X_sliced[:, features]
        return InputData(X=X_sliced, y=y_sliced, groups=g_sliced)


@dataclass
class TrainTestData:
    train_data: InputData
    test_data: InputData


@dataclass
class FeatureRanks:
    def __init__(
        self, features: Iterable[int], ranks: Iterable[float], n_feats: int = None
    ):
        if n_feats is None:
            n_feats = max(features) + 1

        if max(features) >= n_feats:
            raise ValueError("n_feats should be greater than max(features)")
        if max(ranks) > n_feats:
            raise ValueError("ranks should range from 1 to n_feats at most")
        self.n_feats = n_feats
        self._data = dict(zip(features, ranks))

    def __getitem__(self, feature: int):
        if feature >= self.n_feats:
            raise ValueError(
                f"The feature ranks are relative to {self.n_feats} features at most"
            )
        return self._data.get(feature, self.n_feats)


@dataclass
class FeatureEvaluationResults:
    ranks: FeatureRanks
    test_score: float


@dataclass
class OuterLoopResults:
    min_eval: FeatureEvaluationResults
    max_eval: FeatureEvaluationResults
    mid_eval: FeatureEvaluationResults
    score_vs_feats: dict  # TODO: find a better name and signature


@dataclass
class SelectedFeatures:
    min_feats: Iterable[int]
    max_feats: Iterable[int]
    mid_feats: Iterable[int]


@dataclass
class RecursiveFeatureEliminationResults:
    score_vs_feats: dict  # TODO: find a better name and signature
    best_feats: SelectedFeatures


@dataclass
class ScoreCurve:
    n_features: List[int]
    scores: List[float]
