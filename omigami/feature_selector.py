import logging
from typing import Union, List, Dict, Tuple
from concurrent.futures import Executor, Future
import numpy as np
from numpy.random import RandomState
from scipy.stats import gmean

from omigami.data_splitter import DataSplitter
from omigami.data_types import MetricFunction, Estimator, NumpyArray
from omigami.data_models import InputData, SelectedFeatures, FeatureEliminationResults, OuterLoopResults, InnerLoopResults, Split
from omigami.feature_evaluator import FeatureEvaluator
from omigami.post_processor import PostProcessor
from omigami.utils import normalize_score, get_best_n_features, average_ranks

Repetition = List[Union[OuterLoopResults, Future]]


class FeatureSelector:
    def __init__(
        self,
        n_outer: int,
        metric: Union[str, MetricFunction],
        estimator: Union[str, Estimator],
        features_dropout_rate: float = 0.05,
        robust_minimum: float = 0.05,
        n_inner: int = None,
        repetitions: int = 8,
        random_state: int = None,
        executor: Executor = None,
    ):
        self.is_fit = False
        self.random_state = None if random_state is None else RandomState(random_state)
        self.n_outer = n_outer
        self.keep_fraction = 1 - features_dropout_rate
        self.robust_minimum = robust_minimum
        self.repetitions = repetitions
        self.feature_evaluator = FeatureEvaluator(estimator, metric, random_state)

        if not n_inner:
            logging.warning("n_inner is not specified, setting it to n_outer - 1")
            n_inner = n_outer - 1
        self.n_inner = n_inner

        self.n_features = None
        self.selected_features = None
        self.outer_loop_aggregation = None
        self._results = None
        self._minimum_features = 1
        self.post_processor = PostProcessor(robust_minimum)
        self.executor = executor

    def fit(self, X: NumpyArray, y: NumpyArray, groups: NumpyArray = None):
        size = X.shape[0]
        groups = self.get_groups(groups, size)
        input_data = InputData(X=X, y=y, groups=groups)

        # TODO: implement class
        repetition_results = Repetition()

        for _ in range(self.repetitions):
            # TODO: refactor this class
            data_splitter = DataSplitter(
                self.n_outer, self.n_inner, input_data, self.random_state,
            )

            for outer_split in data_splitter.iter_outer_splits():
                outer_loop_results = self._run_outer_loop(
                    input_data, data_splitter, outer_split
                )
                repetition_results.append(outer_loop_results)

        # outer_loop = self._make_outer_loop(input_data)
        # self._results = self._execute_repetitions(outer_loop)
        self.selected_features = self.post_processor.select_features(repetition_results)
        self.is_fit = True
        return self

    def get_groups(self, groups: NumpyArray, size: int):
        if groups is None:
            logging.info("groups is not specified: i.i.d. samples assumed")
            groups = np.arange(size)
        return groups

    def _run_outer_loop(
        self, input_data: InputData, data_splitter: DataSplitter, outer_split: Split
    ) -> OuterLoopResults:

        feature_elimination_results = {}
        feature_set = list(range(input_data.n_features))

        while len(feature_set) >= self._minimum_features:
            inner_results = []

            for inner_split in data_splitter.iter_inner_splits(outer_split):
                inner_loop_data = input_data.split_data(inner_split, feature_set)
                inner_results.append(
                    self.feature_evaluator.evaluate_features(inner_loop_data, feature_set)
                )

            feature_elimination_results[tuple(feature_set)] = inner_results
            feature_set = self._remove_features(feature_set, inner_results)

        outer_loop_results = self.compute_outer_loop_results(
            feature_elimination_results, input_data, outer_split
        )

        return outer_loop_results

    def _remove_features(self, features: NumpyArray, results: InnerLoopResults):
        features_to_keep = np.floor(len(features) * self.keep_fraction)
        features = self._select_n_best(results, features_to_keep)
        return features

    @staticmethod
    def _select_n_best(inner_loop_result: InnerLoopResults, keep_n: int) -> List[int]:
        if keep_n < 1:
            return []
        ranks = [r.ranks for r in inner_loop_result]
        avg_ranks = average_ranks(ranks)
        return get_best_n_features(avg_ranks, keep_n)

    def compute_outer_loop_results(self, feature_elimination_results, input_data, outer_split):
        # TODO: call postprocessor in here and leave the logic there
        n_feats_to_score = self._compute_score_curve(feature_elimination_results)
        best_features = self._select_best_features(feature_elimination_results, n_feats_to_score)
        outer_loop_data_min_feats = input_data.split_data(outer_split, best_features.min_feats)
        outer_loop_data_max_feats = input_data.split_data(outer_split, best_features.max_feats)
        outer_loop_data_mid_feats = input_data.split_data(outer_split, best_features.mid_feats)
        min_eval = self.feature_evaluator.evaluate_features(outer_loop_data_min_feats)
        max_eval = self.feature_evaluator.evaluate_features(outer_loop_data_max_feats)
        mid_eval = self.feature_evaluator.evaluate_features(outer_loop_data_mid_feats)
        outer_loop_results = OuterLoopResults(
            min_eval=min_eval,
            max_eval=max_eval,
            mid_eval=mid_eval,
            score_vs_feats=n_feats_to_score,
        )
        return outer_loop_results

    # def _execute_repetitions(self, outer_loop: OuterLoop) -> List[Repetition]:
    #     results = []
    #     for _ in range(self.repetitions):
    #         outer_loop.refresh_splits()
    #         result = outer_loop.run(executor=self.executor)
    #         results.append(result)
    #     return results
    #
    # def _make_outer_loop(self, input_data: InputData) -> OuterLoop:
    #     feature_evaluator = self._make_feature_evaluator(input_data)
    #     return OuterLoop(
    #         self.n_outer,
    #         feature_evaluator,
    #         self.features_dropout_rate,
    #         self.robust_minimum,
    #     )

    def get_validation_curves(self) -> Dict[str, List]:
        return self.post_processor.get_validation_curves(self._results)

    @staticmethod
    def _compute_score_curve(
        elimination_results: FeatureEliminationResults
    ) -> Dict[int, float]:
        avg_scores = {}
        for features, in_loop_res in elimination_results.items():
            n_feats = len(features)
            test_scores = [r.test_score for r in in_loop_res]
            avg_scores[n_feats] = np.average(test_scores)
        return avg_scores

    def _select_best_features(
        self,
        elimination_results: FeatureEliminationResults,
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
        elimination_results: FeatureEliminationResults
    ) -> Dict[int, Tuple[int]]:
        n_to_features = {}
        for features, in_loop_res in elimination_results.items():
            n_feats = len(features)
            n_to_features[n_feats] = features
        return n_to_features
