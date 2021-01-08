from concurrent.futures._base import Future
from typing import Iterable, List, Dict, Union
from dataclasses import dataclass
from omigami.data import NumpyArray
from omigami.models.estimator import Estimator


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
    min_feats: Union[List[int], NumpyArray, List[str]]
    max_feats: Union[List[int], NumpyArray, List[str]]
    mid_feats: Union[List[int], NumpyArray, List[str]]


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


FeatureSelectionResults = List[List[Union[OuterLoopResults, Future]]]
