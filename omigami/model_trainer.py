from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union
import numpy as np
from scipy.stats import rankdata
import sklearn.metrics
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier


from omigami.utils import NumpyArray, MetricFunction, Estimator


class ModelTrainer:
    """Class to train an estimator across several folds of a given dataset.
    The class creates group splits according to n_inner and n_outer number of CV
    fold. Folds are labelled as (outer_idx, inner_idx (optional)).

    Args:
        estimator (str, BaseEstimator): estimator to be used for feature elimination
        metric (str, callable): metric to be used to assess estimator goodness
        random_state (int): pass an int for reproducible output (default: None)

    """
    RFC = "RFC"

    def __init__(
        self,
        estimator: Estimator,
        metric: MetricFunction,
        random_state: int = None,
    ):
        self.random_state = random_state
        self.estimator = self._make_estimator(estimator)
        self.metric = self._make_metric(metric)

    def evaluate_features(self, X, y, train_split, test_split, features: List[int]) -> TrainingTestingResult:
        """Train and test a clone of self.evaluator over the input split using the
        input features only. It outputs the fitness score over the test split and
        the feature rank of the variables as a TrainingTestingResult object"""

        model = clone(self.estimator)
        y_pred = model.fit(X[train_split], y[train_split]).predict(X[test_split])
        feature_ranks = self._extract_feature_rank(model, features)
        return TrainingTestingResult(
            score=-self.metric(y_pred, y[test_split]), feature_ranks=feature_ranks,
        )

    @staticmethod
    def _extract_feature_rank(
        estimator: Estimator, features: List[int]
    ) -> FeatureRanks:
        """Extract the feature rank from the input estimator. So far it can only handle
        estimators as scikit-learn ones, so either having the `feature_importances_` or
        the `coef_` attribute."""
        if hasattr(estimator, "feature_importances_"):
            ranks = rankdata(-estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            ranks = rankdata(-np.abs(estimator.coef_[0]))
        else:
            raise ValueError("The estimator provided has no feature importances")
        return FeatureRanks(features=features, ranks=ranks)

    def _make_estimator(self, estimator: Union[str, Estimator]) -> Estimator:
        """Make an estimator from the input `estimator`.
        If the estimator is a scikit-learn estimator, then it is simply returned.
        If the estimator is a string then an appropriate estimator corresponding to
        the string is returned. Supported strings are:
            - RFC: Random Forest Classifier
        """
        if estimator == self.RFC:
            return RandomForestClassifier(
                n_estimators=150, n_jobs=-1, random_state=self.random_state
            )
        elif isinstance(estimator, BaseEstimator):
            return estimator
        else:
            raise ValueError(f"Unsupported type of estimator {type(estimator)}.")

    def _make_metric(self, metric: Union[str, MetricFunction]) -> MetricFunction:
        """Build metric function using the input `metric`. If a metric is a string
        then is interpreted as a scikit-learn metric score, such as "accuracy".
        Else, if should be a callable accepting two input arrays."""
        if isinstance(metric, str):
            return self._make_metric_from_string(metric)
        elif hasattr(metric, "__call__"):  # TODO: isinstance(metric, Callable)?
            return metric
        else:
            raise ValueError("Input metric is not valid")

    @staticmethod
    def _make_metric_from_string(metric_string: str) -> MetricFunction:
        if metric_string == "MISS":
            return miss_score
        if metric_string in sklearn.metrics.SCORERS:
            # pylint: disable=protected-access
            return sklearn.metrics.get_scorer(metric_string)._score_func
        raise ValueError("Input metric is not a valid string")


def miss_score(y_true: NumpyArray, y_pred: NumpyArray):
    """MISS score: number of wrong classifications preceded by - so that the higher
    this score the better the model"""
    return -(y_true != y_pred).sum()


@dataclass
class FeatureRanks:
    def __init__(self, features: List[int], ranks: List[float]):
        self.features = features
        self.ranks = ranks
        self._data = dict(zip(features, ranks))

    def __getitem__(self, feature: int) -> float:
        return self._data[feature]

    def __len__(self) -> int:
        return len(self.features)

    def to_dict(self):
        return self._data


@dataclass
class TrainingTestingResult:
    score: float
    feature_ranks: FeatureRanks

    def __getitem__(self, item_name):
        return self.__getattribute__(item_name)
