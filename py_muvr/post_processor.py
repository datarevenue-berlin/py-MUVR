from typing import List, Dict, Tuple
from scipy.stats import gmean
import numpy as np
import pandas as pd
from py_muvr.data_structures import (
    SelectedFeatures,
    ScoreCurve,
    FeatureEliminationResults,
    InnerLoopResults,
    FeatureSelectionRawResults,
    FeatureSelectionResults,
)
from py_muvr.utils import (
    average_ranks,
    average_scores,
    normalize_score,
    get_best_n_features,
)


class PostProcessor:
    """Contains several method that can process the results of the double CV loop
    to perform the actual feature selection and extract data for the plot utils.

    Parameters
    ----------
    robust_minimum : float
        maximum normalized-score value to be considered when computing the `min` and
        `max` selected features
    """

    def __init__(self, robust_minimum: float):
        self.robust_minimum = robust_minimum

    def select_features(self, results: FeatureSelectionRawResults) -> SelectedFeatures:
        """Select the best features for the three possible models (`min`, `mid`
        and `max`).
        The best features are chosen based on their average ranks in the three models
        across repetitions. The number of features are chosen based on the average score
        curve across repetitions and on `robust_minimum`.

        Parameters
        ----------
        results : FeatureSelectionRawResults
            results of the double CV feature selection

        Returns
        -------
        SelectedFeatures
            the best features for the three models
        """
        flat_results = [r for repetition in results for r in repetition]
        average_ranks_min = average_ranks([r.min_eval.ranks for r in flat_results])
        average_ranks_mid = average_ranks([r.mid_eval.ranks for r in flat_results])
        average_ranks_max = average_ranks([r.max_eval.ranks for r in flat_results])
        features_min, features_mid, features_max = self._compute_n_features(results)
        return SelectedFeatures(
            min=get_best_n_features(average_ranks_min, features_min),
            mid=get_best_n_features(average_ranks_mid, features_mid),
            max=get_best_n_features(average_ranks_max, features_max),
        )

    def _compute_n_features(self, results: FeatureSelectionRawResults):
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
    def _get_repetition_avg_scores(results: FeatureSelectionRawResults) -> List:
        avg_scores = []
        for outer_loops_results in results:
            scores = [ol.n_features_to_score_map for ol in outer_loops_results]
            avg_scores.append(average_scores(scores))
        return avg_scores

    def get_validation_curves(
        self, results: FeatureSelectionRawResults
    ) -> Dict[str, List[ScoreCurve]]:
        """Get validation curves (avg score vs number of features) for all outer
        loop iterations, and their average per repetition and overall. Every group
        of curves is stores in a list and packed in an output dictionary:

            {
                "outer_loops": list of outer loop curves,
                "repetitions": list of average ov outer loop curves across repetitions,
                "total": average of all curves,
            }

        Parameters
        ----------
        results : FeatureSelectionRawResults
            results of the double CV feature selection

        Returns
        -------
        Dict[str, List[ScoreCurve]]:
            List of score curves for all outer loop iterations, for all repetitions,
            and overall
        """
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
        """Processes the feature elimination inner loop results to produce
        a score curve and a selection of best features from this curve. Results are
        packed in a FeatureEliminationResults object.

        Parameters
        ----------
        raw_results : Dict[tuple, InnerLoopResults]
            the results of the inner loops for every step of the recursive feature
            elimination

        Returns
        -------
        FeatureEliminationResults
            the processed results of the recursive feature elimination
        """
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
            mid=n_to_features[mid_feats],
            min=n_to_features[min_feats],
            max=n_to_features[max_feats],
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

    def make_average_ranks_df(
        self,
        feature_selection_results: FeatureSelectionResults,
        n_features: int,
        feature_names: List[str] = None,
        exclude_unused_features: bool = True,
    ) -> pd.DataFrame:

        results_df = pd.DataFrame(
            index=np.arange(n_features), columns=["min", "mid", "max"]
        )

        for feature_set in ["min", "mid", "max"]:
            ranks = self._get_feature_ranks(
                feature_selection_results.raw_results, feature_set
            )
            res = pd.DataFrame(ranks).mean()
            results_df.loc[res.index, feature_set] = res.values

        if exclude_unused_features:
            results_df = results_df.dropna(how="all")

        if feature_names is not None:
            results_df.index = [feature_names[i] for i in results_df.index]

        return results_df

    @staticmethod
    def _get_feature_ranks(
        raw_results: FeatureSelectionRawResults, feature_set: str
    ) -> List[Dict[int, float]]:
        ranks = []
        for r in raw_results:
            for ol in r:
                ranks_raw_data = getattr(ol, feature_set + "_eval").ranks.get_data()
                ranks.append(ranks_raw_data)
        return ranks
