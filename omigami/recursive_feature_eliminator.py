from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import gmean
from omigami.feature_evaluator import FeatureEvaluator
from omigami.models import RecursiveFeatureEliminationResults, SelectedFeatures
from omigami.utils import average_ranks, normalize_score, get_best_n_features
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
        avg_scores = self._compute_score_curve(results)
        best_features = self._select_best_features(results, avg_scores)
        return RecursiveFeatureEliminationResults(
            best_feats=best_features,
            score_vs_feats=avg_scores,
        )

    @staticmethod
    def _remove_features(inner_loop_result: InnerLoopResults, keep_n: int) -> List[int]:
        if keep_n < 1:
            return []
        ranks = [r.ranks for r in inner_loop_result]
        avg_ranks = average_ranks(ranks)
        return get_best_n_features(avg_ranks, keep_n)

    @staticmethod
    def _compute_score_curve(elimination_results: Dict[Tuple[int], InnerLoopResults]) -> Dict[int, float]:
        avg_scores = {}
        for features, in_loop_res in elimination_results.items():
            n_feats = len(features)
            test_scores = [r.test_score for r in in_loop_res]
            avg_scores[n_feats] = np.average(test_scores)
        return avg_scores

    @staticmethod
    def _compute_n_features_map(elimination_results: Dict[Tuple[int], InnerLoopResults]) -> Dict[int, Tuple[int]]:
        n_to_features = {}
        for features, in_loop_res in elimination_results.items():
            n_feats = len(features)
            n_to_features[n_feats] = features
        return n_to_features

    def _select_best_features(self, elimination_results: Dict[Tuple[int], InnerLoopResults], avg_scores: Dict[int, float]) -> SelectedFeatures:
        n_to_features = self._compute_n_features_map(elimination_results)
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
