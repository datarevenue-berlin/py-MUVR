# TODO: rename
import numpy as np
from typing import Any
from omigami.types import RandomState
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class Estimator:
    def __init__(self, estimator: Any, random_state: RandomState):
        self._estimator = estimator
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


class ScikitLearnEstimator(Estimator):
    @property
    def feature_importances(self):
        return self._get_feature_importances(self._estimator)

    def _get_feature_importances(self, estimator):
        if hasattr(estimator, "feature_importances_"):
            return estimator.feature_importances_
        if hasattr(estimator, "coef_"):
            return np.abs(estimator.coef_[0])
        if hasattr(estimator, "steps"):
            for _, step in estimator.steps:
                if hasattr(step, "coef_") or hasattr(step, "feature_importances_"):
                    return self._get_feature_importances(step)
        else:
            raise ValueError("The estimator provided has no feature importances")

    def set_random_state(self, random_state: RandomState):
        self._estimator.set_params(random_state=random_state)


def make_estimator(estimator: Any, random_state: RandomState) -> Estimator:
    if isinstance(estimator, str):
        return _make_estimator_from_string(estimator, random_state)
    if isinstance(estimator, BaseEstimator):
        return ScikitLearnEstimator(estimator, random_state)
    raise ValueError("Unknown type of estimator")


def _make_estimator_from_string(estimator: str, random_state: RandomState) -> Estimator:
    if estimator == "RFC":
        rfc = RandomForestClassifier(n_estimators=150)
        return ScikitLearnEstimator(rfc, random_state)
    raise ValueError("Unknown type of estimator")
