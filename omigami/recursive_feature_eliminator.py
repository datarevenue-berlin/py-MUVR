from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from omigami.inner_looper import InnerLooper, InnerLoopResult
from omigami.utils import NumpyArray


class RecursiveFeatureEliminator:
    def __init__(self, features_dropout_rate, min_features):
        self.min_features = min_features
        self.features_dropout_rate = features_dropout_rate
        self.all_features = list(range(self.n_features))

    def run(self, X, y, model_trainer, n_inner, groups: NumpyArray) -> RecursiveFeatureEliminatorResults:
        features = self.all_features
        inner_looper = InnerLooper(n_inner, groups)
        rfe_results = RecursiveFeatureEliminatorResults()

        while len(features) > self.min_features:
            inner_loop_results = inner_looper.run(X[:, features], y, model_trainer)
            rfe_results[features] = inner_loop_results
            features = self._keep_best_features(inner_loop_results, features)

        return rfe_results

    def _keep_best_features(self, inner_cv_results: InnerLoopResult, features) -> List[int]:
        """Keep the best features based on their average rank"""
        feature_ranks = [
            r.feature_ranks.to_dict() for r in inner_cv_results.train_results
        ]
        avg_ranks = pd.DataFrame(feature_ranks).fillna(self.n_features).mean().to_dict()
        sorted_averages = sorted(avg_ranks.items(), key=lambda x: x[1])
        n_feats = len(features)
        n_features_to_drop = round(self.features_dropout_rate * n_feats)
        if not n_features_to_drop:
            n_features_to_drop = 1
        sorted_averages = sorted_averages[:-n_features_to_drop]
        return [feature for feature, _ in sorted_averages]


@dataclass
class RecursiveFeatureEliminatorResults:
    _data: dict = field(default_factory=lambda: {})
    _n_features_map: dict = field(default_factory=lambda: {})
    score: dict = field(default_factory=lambda: {})

    def __setitem__(self, features: List[int], inner_cv_result: InnerLoopResult):
        features = tuple(features)
        self._data[features] = inner_cv_result
        n_features = len(features)
        self._n_features_map[n_features] = features
        self.score[n_features] = inner_cv_result.average_score

    def __iter__(self):
        for item in self._data.items():
            yield item

    def __len__(self):
        return len(self._data)

    def get_features_from_their_number(self, n_features: int) -> List[int]:
        return self._n_features_map[n_features]

    def get_closest_number_of_features(self, n_features: int) -> int:
        return min(self.score.keys(), key=lambda x: abs(x - n_features))
