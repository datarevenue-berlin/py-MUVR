from typing import List
from scipy.stats import gmean
from omigami.outer_loop import OuterLoopResults
from omigami.models import SelectedFeatures
from omigami.utils import average_ranks, average_scores, normalize_score, get_best_n_features

FeatureSelectionResults = List[List[OuterLoopResults]]


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

    def _compute_n_features(self, results: FeatureSelectionResults):
        avg_scores = []
        for outer_loops_results in results:
            scores = [ol.score_vs_feats for ol in outer_loops_results]
            avg_scores.append(average_scores(scores))
        avg_scores = average_scores(avg_scores)
        norm_score = normalize_score(avg_scores)
        n_feats_close_to_min = [n for n, s in norm_score.items() if s <= self.robust_minimum]
        max_feats = max(n_feats_close_to_min)
        min_feats = min(n_feats_close_to_min)
        mid_feats = gmean([max_feats, min_feats])
        mid_feats = min(norm_score.keys(), key=lambda x: abs(x - mid_feats))
        return min_feats, mid_feats, max_feats
