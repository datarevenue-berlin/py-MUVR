# TODO: remove this class, probably saving some tests
from sklearn.metrics import SCORERS, get_scorer
from typing import Union
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from omigami.data_types import MetricFunction, Estimator, RandomState, NumpyArray
from omigami.model import make_estimator


class ModelTrainer:
    def __init__(
        self, estimator: Union[str, Estimator], random_state: Union[int, RandomState]
    ):
        self.random_state = random_state
        self._estimator = make_estimator(estimator, random_state)

    def train_model(self, X: NumpyArray, y: NumpyArray) -> Estimator:
        return self._estimator.train_model(X, y)
