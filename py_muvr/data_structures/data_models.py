from __future__ import annotations

from concurrent.futures._base import Future
from typing import Iterable, List, Dict, Union, Tuple
from dataclasses import dataclass
from py_muvr.data_structures import NumpyArray
from py_muvr.models.estimator import Estimator


@dataclass
class Split:
    id: int
    train_indices: NumpyArray
    test_indices: NumpyArray


@dataclass
class InputDataset:
    X: NumpyArray
    y: NumpyArray
    groups: NumpyArray

    @property
    def n_features(self):
        return self.X.shape[1]

    def __getitem__(self, data_slice: tuple) -> InputDataset:
        rows, features = data_slice
        sliced = InputDataset(
            X=self.X[rows, :], y=self.y[rows], groups=self.groups[rows]
        )
        if features:
            sliced.X = sliced.X[:, features]
        return sliced


@dataclass
class TrainTestData:
    train_data: InputDataset
    test_data: InputDataset


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

    def get_data(self):
        return self._data.copy()


@dataclass
class FeatureEvaluationResults:
    ranks: FeatureRanks
    test_score: float
    model: Estimator


InnerLoopResults = List[FeatureEvaluationResults]


@dataclass
class SelectedFeatures:
    min: Union[List[int], NumpyArray, List[str], Tuple[int]]
    max: Union[List[int], NumpyArray, List[str], Tuple[int]]
    mid: Union[List[int], NumpyArray, List[str], Tuple[int]]

    def __getitem__(self, item):
        accepted_keys = {"min", "mid", "max"}
        if item in accepted_keys:
            return getattr(self, item)
        else:
            raise KeyError(f"Accepted keys are: {accepted_keys}")


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


FeatureSelectionRawResults = List[List[Union[OuterLoopResults, Future]]]


@dataclass
class FeatureSelectionResults:
    raw_results: FeatureSelectionRawResults
    selected_features: SelectedFeatures
    score_curves: Dict[str, List[ScoreCurve]]
    selected_feature_names: SelectedFeatures = None

    def __repr__(self):
        return "FeatureSelectionResults"
