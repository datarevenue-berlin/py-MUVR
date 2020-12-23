import logging
from typing import Callable, Tuple, TypeVar, Union
import numpy as np
from sklearn.base import BaseEstimator

NumpyArray = np.ndarray
MetricFunction = Callable[[NumpyArray, NumpyArray], float]
Split = Tuple[NumpyArray, NumpyArray]
GenericEstimator = TypeVar("GenericEstimator")
Estimator = Union[BaseEstimator, GenericEstimator]


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
        self.random_state = random_state
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
