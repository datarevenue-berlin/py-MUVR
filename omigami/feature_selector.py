import logging
from typing import Union, List, Dict
from concurrent.futures import Executor, Future
import numpy as np
from numpy.random import RandomState

from omigami.data_splitter import DataSplitter
from omigami.recursive_feature_eliminator import RecursiveFeatureEliminator
from omigami.types import MetricFunction, Estimator, NumpyArray
from omigami.models import InputData
from omigami.outer_loop import OuterLoop, OuterLoopResults
from omigami.feature_evaluator import FeatureEvaluator
from omigami.post_processor import PostProcessor


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
        self.features_dropout_rate = features_dropout_rate
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
        self.post_processor = PostProcessor(robust_minimum)
        self.executor = executor

    def fit(self, X: NumpyArray, y: NumpyArray, groups: NumpyArray = None):
        size = X.shape[0]
        groups = self.get_groups(groups, size)
        input_data = InputData(X=X, y=y, groups=groups)

        # TODO: implement class
        aggregator = ResultsAggregator()  # TODO: could this be an attribute?

        for r in range(self.repetitions):
            # TODO: refactor this class
            data_splitter = DataSplitter(size, self.n_outer, self.n_inner, r)

            for outer_split in data_splitter.outer_splits:
                # TODO: possibly implement it like this to make signature simpler
                outer_loop_data = input_data.split(outer_split)

                # TODO: implement this method. Outer loop results can be a dataclass that
                # TODO: is expected by add_outer_loop_results
                outer_loop_results = self._run_outer_loop(outer_loop_data, data_splitter)
                aggregator.add_outer_loop_results(outer_loop_results)

        # outer_loop = self._make_outer_loop(input_data)
        # self._results = self._execute_repetitions(outer_loop)
        # self.selected_features = self.post_processor.select_features(self._results)
        # self.is_fit = True
        return self

    def _run_outer_loop(self, outer_loop_data, data_splitter):
        # TODO: refactor this class as needed
        rfe = RecursiveFeatureEliminator(outer_loop_data.n_features)

        # TODO: need implementation
        feature_elimination_results = FeatureEliminationResults()

        for feature_set in rfe.eliminate_features():
            inner_results = []
            for inner_split in data_splitter.inner_splits:
                inner_loop_data = outer_loop_data.split(inner_split, feature_set)
                inner_results.append(self.feature_evaluator.evaluate_features(
                    inner_loop_data
                ))
            feature_elimination_results.add(inner_results)

        return feature_elimination_results

    def get_groups(self, groups, size):
        if groups is None:
            logging.info("groups is not specified: i.i.d. samples assumed")
            groups = np.arange(size)
        return groups

    def _execute_repetitions(self, outer_loop: OuterLoop) -> List[Repetition]:
        results = []
        for _ in range(self.repetitions):
            outer_loop.refresh_splits()
            result = outer_loop.run(executor=self.executor)
            results.append(result)
        return results

    def _make_outer_loop(self, input_data: InputData) -> OuterLoop:
        feature_evaluator = self._make_feature_evaluator(input_data)
        return OuterLoop(
            self.n_outer,
            feature_evaluator,
            self.features_dropout_rate,
            self.robust_minimum,
        )

    def get_validation_curves(self) -> Dict[str, List]:
        return self.post_processor.get_validation_curves(self._results)
