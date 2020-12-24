import logging
from typing import Union
import numpy as np
from omigami.types import MetricFunction, Estimator
from omigami.models import InputData
from omigami.data_splitter import DataSplitter


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
        self.results = None

    def fit(self, X, y, groups=None):
        if groups is None:
            logging.info("groups is not specified: i.i.d. samples assumed")
            groups = np.arange(X.shape[0])
        input_data = InputData(X=X, y=y, groups=groups)
        data_splitter = DataSplitter(
            self.n_outer, self.n_outer, input_data, random_state=self.random_state
        )

        self.n_features = X.shape[1]

        return True
