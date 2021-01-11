from __future__ import annotations

import logging
from concurrent.futures import Executor, Future
from typing import Union, List, Dict, Tuple

import numpy as np
from numpy.random import RandomState

from omigami.data_structures import (
    InputDataset,
    SelectedFeatures,
    OuterLoopResults,
    InnerLoopResults,
    Split,
    FeatureEvaluationResults,
    MetricFunction,
    InputEstimator,
    NumpyArray,
    FeatureSelectionResults,
)
from omigami.feature_evaluator import FeatureEvaluator
from omigami.data_splitter import DataSplitter
from omigami.post_processor import PostProcessor
from omigami.utils import get_best_n_features, average_ranks

Repetition = List[Union[OuterLoopResults, Future]]


class FeatureSelector:
    def __init__(
        self,
        n_outer: int,
        metric: Union[str, MetricFunction],
        estimator: Union[str, InputEstimator],
        features_dropout_rate: float = 0.05,
        robust_minimum: float = 0.05,
        n_inner: int = None,
        n_repetitions: int = 8,
        random_state: int = None,
    ):
        self.is_fit = False
        self.random_state = None if random_state is None else RandomState(random_state)
        self.n_outer = n_outer
        self.keep_fraction = 1 - features_dropout_rate
        self.n_repetitions = n_repetitions
        self.feature_evaluator = FeatureEvaluator(estimator, metric, random_state)

        if not n_inner:
            logging.info("n_inner is not specified, setting it to n_outer - 1")
            n_inner = n_outer - 1
        self.n_inner = n_inner

        self._selected_features = None
        self.outer_loop_aggregation = None
        self.results = None
        self._minimum_features = 1
        self.post_processor = PostProcessor(robust_minimum)

    def fit(
        self,
        X: NumpyArray,
        y: NumpyArray,
        groups: NumpyArray = None,
        executor: Executor = None,
    ) -> FeatureSelector:
        size, n_features = X.shape
        groups = self._get_groups(groups, size)
        input_data = InputDataset(X=X, y=y, groups=groups)
        self.feature_evaluator.set_n_initial_features(n_features)

        repetition_results = []

        for _ in range(self.n_repetitions):
            data_splitter = DataSplitter(
                self.n_outer, self.n_inner, input_data, self.random_state,
            )

            outer_loop_results = []
            for outer_split in data_splitter.iter_outer_splits():
                outer_loop_result = self._deferred_run_outer_loop(
                    input_data,
                    outer_split,
                    executor=executor,
                    data_splitter=data_splitter,
                )
                outer_loop_results.append(outer_loop_result)
            repetition_results.append(outer_loop_results)

        self._selected_features = self._select_best_features(repetition_results)
        self.is_fit = True
        return self

    @staticmethod
    def _get_groups(groups: NumpyArray, size: int) -> NumpyArray:
        if groups is None:
            logging.info("groups is not specified: i.i.d. samples assumed")
            groups = np.arange(size)
        return groups

    def _deferred_run_outer_loop(
        self,
        input_data: InputDataset,
        outer_split: Split,
        data_splitter: DataSplitter,
        executor: Executor,
    ) -> Union[Future, OuterLoopResults]:
        if executor is None:
            return self._run_outer_loop(input_data, outer_split, data_splitter)
        return executor.submit(
            self._run_outer_loop, input_data, outer_split, data_splitter
        )

    def _run_outer_loop(
        self, input_data: InputDataset, outer_split: Split, data_splitter: DataSplitter,
    ) -> OuterLoopResults:

        raw_feature_elim_results = {}
        feature_set = list(range(input_data.n_features))

        while len(feature_set) >= self._minimum_features:
            inner_results = []

            for inner_split in data_splitter.iter_inner_splits(outer_split):
                inner_loop_data = data_splitter.split_data(
                    input_data, inner_split, feature_set
                )

                feature_evaluation_results = self.feature_evaluator.evaluate_features(
                    inner_loop_data, feature_set
                )

                inner_results.append(feature_evaluation_results)

            raw_feature_elim_results[tuple(feature_set)] = inner_results
            feature_set = self._remove_features(feature_set, inner_results)

        outer_loop_results = self._create_outer_loop_results(
            raw_feature_elim_results, input_data, outer_split, data_splitter
        )

        return outer_loop_results

    def _remove_features(
        self, features: List[int], results: InnerLoopResults
    ) -> List[int]:
        features_to_keep = int(np.floor(len(features) * self.keep_fraction))
        features = self._select_n_best(results, features_to_keep)
        return features

    @staticmethod
    def _select_n_best(inner_loop_result: InnerLoopResults, keep_n: int) -> List[int]:
        if keep_n < 1:
            return []
        ranks = [r.ranks for r in inner_loop_result]
        avg_ranks = average_ranks(ranks)
        return get_best_n_features(avg_ranks, keep_n)

    def _create_outer_loop_results(
        self,
        raw_feature_elim_results: Dict[tuple, InnerLoopResults],
        input_data: InputDataset,
        outer_split: Split,
        data_splitter: DataSplitter,
    ) -> OuterLoopResults:
        feature_elimination_results = self.post_processor.process_feature_elim_results(
            raw_feature_elim_results
        )
        min_eval, mid_eval, max_eval = self._evaluate_min_mid_and_max_features(
            input_data,
            feature_elimination_results.best_features,
            outer_split,
            data_splitter,
        )
        outer_loop_results = OuterLoopResults(
            min_eval=min_eval,
            mid_eval=mid_eval,
            max_eval=max_eval,
            n_features_to_score_map=feature_elimination_results.n_features_to_score_map,
        )
        return outer_loop_results

    def _evaluate_min_mid_and_max_features(
        self,
        input_data: InputDataset,
        best_features: SelectedFeatures,
        split: Split,
        data_splitter: DataSplitter,
    ) -> Tuple[
        FeatureEvaluationResults, FeatureEvaluationResults, FeatureEvaluationResults
    ]:
        min_feats = best_features.min_feats
        mid_feats = best_features.mid_feats
        max_feats = best_features.max_feats

        data_min_feats = data_splitter.split_data(input_data, split, min_feats)
        data_mid_feats = data_splitter.split_data(input_data, split, mid_feats)
        data_max_feats = data_splitter.split_data(input_data, split, max_feats)

        min_eval = self.feature_evaluator.evaluate_features(data_min_feats, min_feats)
        mid_eval = self.feature_evaluator.evaluate_features(data_mid_feats, mid_feats)
        max_eval = self.feature_evaluator.evaluate_features(data_max_feats, max_feats)

        return min_eval, mid_eval, max_eval

    def _select_best_features(
        self, repetition_results: FeatureSelectionResults
    ) -> SelectedFeatures:
        self.results = self.post_processor.fetch_results(repetition_results)
        selected_features = self.post_processor.select_features(self.results)
        return selected_features

    def get_validation_curves(self) -> Dict[str, List]:
        return self.post_processor.get_validation_curves(self.results)

    def get_selected_features(self, feature_names: List[str] = None):

        if not self.is_fit:
            # TODO: custom exception
            raise RuntimeError("The feature selector is not fit yet")

        if feature_names is not None:
            min_names = [feature_names[f] for f in self._selected_features.min_feats]
            mid_names = [feature_names[f] for f in self._selected_features.mid_feats]
            max_names = [feature_names[f] for f in self._selected_features.max_feats]

            return SelectedFeatures(
                min_feats=min_names, max_feats=max_names, mid_feats=mid_names,
            )

        # TODO: in one case we return a "copy" in this case we return a reference.
        # maybe it's better to always return a copy
        return self._selected_features
