import logging
from typing import Union
import numpy as np
from omigami.types import MetricFunction, Estimator
from omigami.models import InputData
from omigami.outer_loop import OuterLoop
from omigami.feature_evaluator import FeatureEvaluator
from omigami.post_processor import PostProcessor


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
    ):
        self.is_fit = False
        self.random_state = np.random.RandomState(random_state)
        self.n_outer = n_outer
        self.metric = metric
        self.estimator = estimator
        self.features_dropout_rate = features_dropout_rate
        self.robust_minimum = robust_minimum
        self.repetitions = repetitions

        if not n_inner:
            logging.warning("n_inner is not specified, setting it to n_outer - 1")
            n_inner = n_outer - 1
        self.n_inner = n_inner

        self.n_features = None
        self.selected_features = None
        self.outer_loop_aggregation = None
        self._results = None
        self.post_processor = PostProcessor(robust_minimum)

    def fit(self, X, y, groups=None):
        if groups is None:
            logging.info("groups is not specified: i.i.d. samples assumed")
            groups = np.arange(X.shape[0])
        input_data = InputData(X=X, y=y, groups=groups)
        outer_loop = self._make_outer_loop(input_data)
        self._results = self._execute_repetitions(outer_loop)
        self.selected_features = self.post_processor.select_features(self._results)
        return self

    def _execute_repetitions(self, outer_loop):
        results = []
        for _ in range(self.repetitions):
            outer_loop.refresh_splits()
            result = outer_loop.run()
            results.append(result)
        return results

    def _make_feature_evaluator(self, input_data):
        return FeatureEvaluator(
            input_data,
            self.n_outer,
            self.n_inner,
            self.estimator,
            self.metric,
            self.random_state,
        )

    def _make_outer_loop(self, input_data):
        feature_evaluator = self._make_feature_evaluator(self, input_data)
        return OuterLoop(
            self.n_outer,
            feature_evaluator,
            self.features_dropout_rate,
            self.robust_minimum,
        )
