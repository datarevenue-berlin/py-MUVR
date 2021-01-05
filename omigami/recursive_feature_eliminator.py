from typing import List
import numpy as np
from omigami.data_models import InnerLoopResults
from omigami.utils import average_ranks, get_best_n_features


class RecursiveFeatureEliminator:
    def __init__(
        self,
        dropout_rate: float,
        n_features: int
    ):
        self.keep_fraction = 1 - dropout_rate
        self._n_features = n_features
        self.features = np.arange(self._n_features)

    def iter_features(self):
        while self.stop_condition(self.features):
            yield self.features

    @staticmethod
    def stop_condition(features):
        return len(features) >= 1

    def remove_features(self, results: InnerLoopResults):
        features_to_keep = np.floor(len(self.features) * self.keep_fraction)
        features = self._remove_features(results, features_to_keep)
        self.features = features

    @staticmethod
    def _remove_features(inner_loop_result: InnerLoopResults, keep_n: int) -> List[int]:
        if keep_n < 1:
            return []
        ranks = [r.ranks for r in inner_loop_result]
        avg_ranks = average_ranks(ranks)
        return get_best_n_features(avg_ranks, keep_n)
