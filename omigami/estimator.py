# TODO: rename estimator.py --> model_trainer.py
from sklearn.metrics import SCORERS, get_scorer
from typing import Union
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from omigami.types import MetricFunction, Estimator, RandomState, NumpyArray
from model import make_estimator


class ModelTrainer:

    RFC = "RFC"

    def __init__(
        self, estimator: Union[str, Estimator], random_state: Union[int, RandomState]
    ):
        self.random_state = random_state
        self._estimator = make_estimator(estimator, random_state)

    def train_model(self, X: NumpyArray, y: NumpyArray) -> Estimator:
        estimator = clone(self._estimator)  # refresh in case fit has memory
        estimator.fit(X, y)
        return estimator
