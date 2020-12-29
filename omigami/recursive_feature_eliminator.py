from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import gmean
from omigami.feature_evaluator import FeatureEvaluator
from omigami.models import RecursiveFeatureEliminationResults, SelectedFeatures
from omigami.utils import average_ranks, normalize_score
from omigami.inner_loop import InnerLoop, InnerLoopResults


class RecursiveFeatureEliminator:
    def __init__(self, feature_evaluator: FeatureEvaluator, dropout_rate: float, robust_minimum: float):
        self.keep_fraction = 1 - dropout_rate
        self.robust_minimum = robust_minimum
        self.inner_loop = InnerLoop(feature_evaluator)
        self._n_features = feature_evaluator.get_n_features()

    def run(self, outer_loop_index):
        results = {}
        features = np.arange(self._n_features)
        while len(features) >= 1:
            inner_loop_result = self.inner_loop.run(features, outer_loop_index)
            results[tuple(features)] = inner_loop_result
            features_to_keep = np.floor(len(features) * self.keep_fraction)
            features = self._remove_features(inner_loop_result, features_to_keep)
        best_features = self._select_best_features(results)
        return RecursiveFeatureEliminationResults(
            best_feats=best_features,
            score_vs_feats=results,
        )

    @staticmethod
    def _remove_features(inner_loop_result: InnerLoopResults, keep_n: int) -> List[int]:
        if keep_n < 1:
            return []
        ranks = [r.ranks for r in inner_loop_result]
        avg_ranks = average_ranks(ranks)
        # TODO: improve efficiency of range(avg_ranks.n_feats including an items
        # method to FeatureRanks so that we don't have to go through all n_feats
        # (those that are not there are useless, since they are the highest ranks)
        best_ranks = sorted([(avg_ranks[f], f) for f in range(avg_ranks.n_feats)])
        best_ranks = best_ranks[:int(keep_n)]
        return [f for _, f in best_ranks]

    def _select_best_features(self, elimination_results: Dict[Tuple[int], InnerLoopResults]) -> SelectedFeatures:
        avg_scores = {}
        n_to_features = {}
        for features, in_loop_res in elimination_results.items():
            n_feats = len(features)
            n_to_features[n_feats] = features
            test_scores = [r.test_score for r in in_loop_res]
            avg_scores[n_feats] = np.average(test_scores)
        norm_score = normalize_score(avg_scores)
        n_feats_close_to_min = [n for n, s in norm_score.items() if s <= self.robust_minimum]
        max_feats = max(n_feats_close_to_min)
        min_feats = min(n_feats_close_to_min)
        mid_feats = gmean([max_feats, min_feats])
        mid_feats = min(avg_scores.keys(), key=lambda x: abs(x - mid_feats))

        return SelectedFeatures(
            mid_feats=n_to_features[mid_feats],
            min_feats=n_to_features[min_feats],
            max_feats=n_to_features[max_feats],
        )


