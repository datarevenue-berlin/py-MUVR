# TODO: rename
import numpy as np
from typing import Any
from omigami.types import RandomState
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


class Estimator:
    def __init__(self, estimator: Any, random_state: RandomState):
        self._estimator = estimator
        self._random_state = random_state
        self.set_random_state(random_state)

    def fit(self, X, y):
        self._estimator.fit(X, y)
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    @property
    def feature_importances(self):
        raise NotImplementedError("feature_importance is not implemented")

    def set_random_state(self, random_state: RandomState):
        raise NotImplementedError("set_random_state is not implemented")

    def _clone_estimator(self):
        raise NotImplementedError("_clone_estimator is not implemented")

    def clone(self):
        estimator_clone = self._clone_estimator()
        return self.__class__(estimator_clone, self._random_state)

    def train_model(self, X, y):
        return self.clone().fit(X, y)


class ScikitLearnEstimator(Estimator):
    @property
    def feature_importances(self):
        if hasattr(self._estimator, "feature_importances_"):
            return self._estimator.feature_importances_
        if hasattr(self._estimator, "coef_"):
            return np.abs(self._estimator.coef_[0])
        raise ValueError("The estimator provided has no feature importances")

    def set_random_state(self, random_state: RandomState):
        self._estimator.set_params(random_state=random_state)

    def _clone_estimator(self):
        return clone(self._estimator)


class ScikitLearnPipeline(Estimator):
    @property
    def feature_importances(self):
        for _, step in self._estimator:
            if hasattr(step, "feature_importances_"):
                return step.feature_importances_
            if hasattr(step, "coef_"):
                return np.abs(step.coef_[0])
        raise ValueError("The estimator provided has no feature importances")

    def set_random_state(self, random_state: RandomState):
        for _, step in self._estimator.steps:
            try:
                step.set_params(random_state=random_state)
            except ValueError:
                pass  # not all the elements of the pipeline have to be random

    def _clone_estimator(self):
        return clone(self._estimator)


def make_estimator(estimator: Any, random_state: RandomState) -> Estimator:
    if isinstance(estimator, str):
        return _make_estimator_from_string(estimator, random_state)
    if isinstance(estimator, Pipeline):
        return ScikitLearnPipeline(estimator, random_state)
    if isinstance(estimator, BaseEstimator):
        return ScikitLearnEstimator(estimator, random_state)
    raise ValueError("Unknown type of estimator")


def _make_estimator_from_string(estimator: str, random_state: RandomState) -> Estimator:
    if estimator == "RFC":
        rfc = RandomForestClassifier(n_estimators=150)
        return ScikitLearnEstimator(rfc, random_state)
    raise ValueError("Unknown type of estimator")
