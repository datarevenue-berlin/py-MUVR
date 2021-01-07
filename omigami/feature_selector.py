import logging
from typing import Union, List, Dict, Tuple
from concurrent.futures import Executor, Future
import numpy as np
from numpy.random import RandomState
from scipy.stats import gmean

from omigami.data_models import FeatureEvaluationResults
from omigami.data_splitter import DataSplitter
from omigami.data_types import MetricFunction, Estimator, NumpyArray
from omigami.data_models import (
    InputData,
    SelectedFeatures,
    FeatureEliminationResults,
    OuterLoopResults,
    InnerLoopResults,
    Split,
)
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
        size, n_features = X.shape[0], X.shape[1]
        groups = self.get_groups(groups, size)
        input_data = InputData(X=X, y=y, groups=groups, n_features=n_features)
        self.feature_evaluator.n_initial_features = n_features

        repetition_results = []

        for _ in range(self.repetitions):
            data_splitter = DataSplitter(
                self.n_outer, self.n_inner, input_data, self.random_state,
            )
            olrs = []
            for outer_split in data_splitter.iter_outer_splits():
                outer_loop_results = self._run_outer_loop(
                    input_data, data_splitter, outer_split
                )
                olrs.append(outer_loop_results)
            repetition_results.append(olrs)
        # outer_loop = self._make_outer_loop(input_data)
        # self._results = self._execute_repetitions(outer_loop)
        self._results = repetition_results
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

        raw_feature_elim_results = {}
        feature_set = list(range(input_data.n_features))

        while len(feature_set) >= self._minimum_features:
            inner_results = []

            for inner_split in data_splitter.iter_inner_splits(outer_split):
                inner_loop_data = input_data.split_data(inner_split, feature_set)
                inner_results.append(
                    self.feature_evaluator.evaluate_features(
                        inner_loop_data, feature_set
                    )
                )

            raw_feature_elim_results[tuple(feature_set)] = inner_results
            feature_set = self._remove_features(feature_set, inner_results)

        feature_elimination_results = self.post_processor.process_feature_elim_results(
            raw_feature_elim_results
        )
        outer_loop_results = self.create_outer_loop_results(
            feature_elimination_results, input_data, outer_split
        )

        return outer_loop_results

    def _remove_features(self, features: List[int], results: InnerLoopResults):
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

    def create_outer_loop_results(
        self,
        feature_elimination_results: FeatureEliminationResults,
        input_data: InputData,
        outer_split: Split,
    ) -> OuterLoopResults:
        min_eval, mid_eval, max_eval = self.evaluate_min_mid_and_max_features(
            input_data, feature_elimination_results.best_features, outer_split,
        )
        outer_loop_results = OuterLoopResults(
            min_eval=min_eval,
            mid_eval=mid_eval,
            max_eval=max_eval,
            n_features_to_score_map=feature_elimination_results.n_features_to_score_map,
        )
        return outer_loop_results

    def evaluate_min_mid_and_max_features(
        self, input_data: InputData, best_features: SelectedFeatures, split: Split,
    ) -> Tuple[
        FeatureEvaluationResults, FeatureEvaluationResults, FeatureEvaluationResults
    ]:
        outer_loop_data_min_feats = input_data.split_data(split)
        outer_loop_data_max_feats = input_data.split_data(split)
        outer_loop_data_mid_feats = input_data.split_data(split)
        min_eval = self.feature_evaluator.evaluate_features(
            outer_loop_data_min_feats, best_features.min_feats
        )
        mid_eval = self.feature_evaluator.evaluate_features(
            outer_loop_data_mid_feats, best_features.mid_feats
        )
        max_eval = self.feature_evaluator.evaluate_features(
            outer_loop_data_max_feats, best_features.max_feats
        )
        return min_eval, mid_eval, max_eval

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
