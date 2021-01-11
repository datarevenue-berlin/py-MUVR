from typing import List, Dict, Tuple
from scipy.stats import gmean
import numpy as np
from omigami.data_structures import (
    SelectedFeatures,
    ScoreCurve,
    FeatureEliminationResults,
    InnerLoopResults,
    FeatureSelectionResults,
)
from omigami.utils import (
    average_ranks,
    average_scores,
    normalize_score,
    get_best_n_features,
)


class PostProcessor:
    def __init__(self, robust_minimum):
        self.robust_minimum = robust_minimum

    def select_features(self, results: FeatureSelectionResults) -> SelectedFeatures:
        flat_results = [r for repetition in results for r in repetition]
        average_ranks_min = average_ranks([r.min_eval.ranks for r in flat_results])
        average_ranks_mid = average_ranks([r.mid_eval.ranks for r in flat_results])
        average_ranks_max = average_ranks([r.max_eval.ranks for r in flat_results])
        features_min, features_mid, features_max = self._compute_n_features(results)
        return SelectedFeatures(
            min_feats=get_best_n_features(average_ranks_min, features_min),
            mid_feats=get_best_n_features(average_ranks_mid, features_mid),
            max_feats=get_best_n_features(average_ranks_max, features_max),
        )

    @staticmethod
    def fetch_results(results: FeatureSelectionResults) -> FeatureSelectionResults:
        fetched_results = []
        for repetition in results:
            fetched_repetition = [ol.result() for ol in repetition]
            fetched_results.append(fetched_repetition)
        return fetched_results

    def _compute_n_features(self, results: FeatureSelectionResults):
        avg_rep_scores = self._get_repetition_avg_scores(results)
        avg_scores = average_scores(avg_rep_scores)
        norm_score = normalize_score(avg_scores)
        n_feats_close_to_min = [
            n for n, s in norm_score.items() if s <= self.robust_minimum
        ]
        max_feats = max(n_feats_close_to_min)
        min_feats = min(n_feats_close_to_min)
        mid_feats = gmean([max_feats, min_feats])
        mid_feats = min(norm_score.keys(), key=lambda x: abs(x - mid_feats))
        return min_feats, mid_feats, max_feats

    @staticmethod
    def _get_repetition_avg_scores(results: FeatureSelectionResults) -> List:
        avg_scores = []
        for outer_loops_results in results:
            scores = [ol.n_features_to_score_map for ol in outer_loops_results]
            avg_scores.append(average_scores(scores))
        return avg_scores

    def get_validation_curves(self, results: FeatureSelectionResults) -> Dict:
        flat_results = [r for repetition in results for r in repetition]
        outer_loop_scores = [r.n_features_to_score_map for r in flat_results]
        avg_scores_per_loop = self._get_repetition_avg_scores(results)
        avg_scores = average_scores(avg_scores_per_loop)
        return {
            "outer_loops": [self._score_to_score_curve(s) for s in outer_loop_scores],
            "repetitions": [self._score_to_score_curve(s) for s in avg_scores_per_loop],
            "total": [self._score_to_score_curve(avg_scores)],
        }

    @staticmethod
    def _score_to_score_curve(scores: Dict[int, float]) -> ScoreCurve:
        n_features, score_values = zip(*sorted(scores.items()))
        return ScoreCurve(n_features=n_features, scores=score_values)

    def process_feature_elim_results(self, raw_results: Dict[tuple, InnerLoopResults]):
        n_feats_to_score = self._compute_score_curve(raw_results)
        best_features = self._select_best_outer_features(raw_results, n_feats_to_score)

        return FeatureEliminationResults(n_feats_to_score, best_features)

    @staticmethod
    def _compute_score_curve(
        elimination_results: Dict[tuple, InnerLoopResults]
    ) -> Dict[int, float]:
        avg_scores = {}
        for features, in_loop_res in elimination_results.items():
            n_feats = len(features)
            test_scores = [r.test_score for r in in_loop_res]
            avg_scores[n_feats] = np.average(test_scores)
        return avg_scores

    def _select_best_outer_features(
        self,
        elimination_results: Dict[tuple, InnerLoopResults],
        avg_scores: Dict[int, float],
    ) -> SelectedFeatures:
        n_to_features = self._compute_n_features_map(elimination_results)
        norm_score = normalize_score(avg_scores)
        n_feats_close_to_min = [
            n for n, s in norm_score.items() if s <= self.robust_minimum
        ]
        max_feats = max(n_feats_close_to_min)
        min_feats = min(n_feats_close_to_min)
        mid_feats = gmean([max_feats, min_feats])
        mid_feats = min(avg_scores.keys(), key=lambda x: abs(x - mid_feats))

        return SelectedFeatures(
            mid_feats=n_to_features[mid_feats],
            min_feats=n_to_features[min_feats],
            max_feats=n_to_features[max_feats],
        )

    @staticmethod
    def _compute_n_features_map(
        elimination_results: Dict[tuple, InnerLoopResults]
    ) -> Dict[int, Tuple[int]]:
        n_to_features = {}
        for features in elimination_results.keys():
            n_feats = len(features)
            n_to_features[n_feats] = features
        return n_to_features
