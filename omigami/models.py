# TODO: rename as data_models

from typing import Iterable, Union
from dataclasses import dataclass
from omigami.types import NumpyArray


# makes sense to me to group these three objects into
# one, although I'm not sure it's not overkill
@dataclass
class InputData:
    X: NumpyArray
    y: NumpyArray
    groups: NumpyArray


@dataclass
class FeatureRanks:
    def __init__(self, features: Iterable[int], ranks: Iterable[float], n_feats: int = None):
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
