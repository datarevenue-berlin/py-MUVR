# TODO: rename estimator.py --> model_trainer.py
from sklearn.metrics import SCORERS, get_scorer
from typing import Union
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from omigami.types import MetricFunction, Estimator, RandomState, NumpyArray


class ModelTrainer:

    RFC = "RFC"

    def __init__(
        self, estimator: Union[str, Estimator], random_state: Union[int, RandomState]
    ):
        self.random_state = random_state
        self._estimator = self._make_estimator(estimator)

    def _make_estimator(self, estimator: Union[str, Estimator]) -> Estimator:
        """Make an estimator from the input `estimator`.
        If the estimator is a scikit-learn estimator, then it is simply returned.
        If the estimator is a string then an appropriate estimator corresponding to
        the string is returned. Supported strings are:
            - RFC: Random Forest Classifier
        """
        if estimator == self.RFC:
            return RandomForestClassifier(
                n_estimators=150, random_state=self.random_state
            )
        elif isinstance(estimator, BaseEstimator):
            return estimator
        else:
            raise ValueError("Unknown type of estimator")

    def train_model(self, X: NumpyArray, y: NumpyArray) -> Estimator:
        estimator = clone(self._estimator)  # refresh in case fit has memory
        estimator.fit(X, y)
        return estimator


def miss_score(y_true: NumpyArray, y_pred: NumpyArray):
    """MISS score: number of wrong classifications preceded by - so that the higher
    this score the better the model"""
    return -(y_true != y_pred).sum()
