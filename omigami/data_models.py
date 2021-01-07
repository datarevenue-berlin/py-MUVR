# TODO: rename as data_models

from typing import Iterable, List, Dict, Union
from dataclasses import dataclass
from omigami.data_types import NumpyArray
from omigami.model import Estimator


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
    n_features: int = None

    def __post_init__(self):
        if self.n_features is None:
            self.n_features = self.X.shape[1]

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
    model: Estimator


InnerLoopResults = List[FeatureEvaluationResults]


@dataclass
class SelectedFeatures:
    min_feats: Union[List[int], NumpyArray]
    max_feats: Union[List[int], NumpyArray]
    mid_feats: Union[List[int], NumpyArray]


@dataclass
class FeatureEliminationResults:
    n_features_to_score_map: Dict[int, float]
    best_features: SelectedFeatures


@dataclass
class OuterLoopResults:
    min_eval: FeatureEvaluationResults
    max_eval: FeatureEvaluationResults
    mid_eval: FeatureEvaluationResults
    n_features_to_score_map: Dict[int, float]


@dataclass
class ScoreCurve:
    n_features: List[int]
    scores: List[float]
