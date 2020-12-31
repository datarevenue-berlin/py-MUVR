from typing import List, Union, Dict
from concurrent.futures import Future
from scipy.stats import gmean
from omigami.outer_loop import OuterLoopResults
from omigami.models import SelectedFeatures, ScoreCurve
from omigami.utils import (
    average_ranks,
    average_scores,
    normalize_score,
    get_best_n_features,
)

FeatureSelectionResults = List[List[Union[OuterLoopResults, Future]]]


class PostProcessor:
    def __init__(self, robust_minimum):
        self.robust_minimum = robust_minimum

    def select_features(self, results: FeatureSelectionResults) -> SelectedFeatures:
        results = self._fetch_results(results)
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
    def _fetch_results(results: FeatureSelectionResults) -> FeatureSelectionResults:
        fetched_results = []
        for repetition in results:
            fetched_repetition = []
            for outer_iteration in repetition:
                if isinstance(outer_iteration, Future):
                    outer_iteration = outer_iteration.result()
                fetched_repetition.append(outer_iteration)
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
            scores = [ol.score_vs_feats for ol in outer_loops_results]
            avg_scores.append(average_scores(scores))
        return avg_scores

    def get_validation_curves(self, results: FeatureSelectionResults) -> Dict:
        results = self._fetch_results(results)
        flat_results = [r for repetition in results for r in repetition]
        outer_loop_scores = [r.score_vs_feats for r in flat_results]
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
