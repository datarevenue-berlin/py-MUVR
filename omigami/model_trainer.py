from typing import Callable, Dict, List, Tuple, TypeVar, Union
import numpy as np
from scipy.stats import rankdata
import sklearn.metrics
from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier

NumpyArray = np.ndarray
MetricFunction = Callable[[NumpyArray, NumpyArray], float]
Split = Tuple[NumpyArray, NumpyArray]
GenericEstimator = TypeVar("GenericEstimator")
Estimator = Union[BaseEstimator, GenericEstimator]


class ModelTrainer:

    RFC = "RFC"

    def __init__(
        self, X, y, groups, n_inner, n_outer, estimator, metric, random_state=None
    ):
        self.random_state = random_state
        self.X = X
        self.n_features = X.shape[1]
        self.y = y
        self.groups = groups
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.splits = self._make_splits()
        self.estimator = self._make_estimator(estimator)
        self.metric = self._make_metric(metric)

    def _make_splits(self) -> Dict[tuple, Split]:
        """Create a dictionary of split indexes for self.X and self.y,
         according to self.n_outer and self.n_inner and self.groups.
        The groups are split first in n_outer test and train segments. Then each
        train segment is split in n_inner smaller test and train sub-segments.
        The splits are keyed (outer_index_split, n_inner_split).
        Outer splits are simply keyed (outer_index_split,).
        """
        outer_splitter = GroupKFold(self.n_outer)
        inner_splitter = GroupKFold(self.n_inner)
        outer_splits = outer_splitter.split(self.X, self.y, self.groups)
        splits = {}
        for i, (out_train, out_test) in enumerate(outer_splits):
            splits[(i,)] = out_train, out_test
            inner_splits = inner_splitter.split(
                self.X[out_train, :], self.y[out_train], self.groups[out_train]
            )
            for j, (inner_train, inner_valid) in enumerate(inner_splits):
                splits[(i, j)] = out_train[inner_train], out_train[inner_valid]
        return splits

    def run(self, split_id, features: List[int]) -> Dict:
        """Train and test a clone of self.evaluator over the input split using the
        input features only. It outputs the fitness score over the test split and
        the feature rank of the variables"""
        inner_train_idx, inner_test_idx = self.splits[tuple(split_id)]
        X_train = self.X[inner_train_idx, :][:, features]
        X_test = self.X[inner_test_idx, :][:, features]
        y_train = self.y[inner_train_idx]
        y_test = self.y[inner_test_idx]
        model = clone(self.estimator)
        y_pred = model.fit(X_train, y_train).predict(X_test)
        feature_ranks = self._extract_feature_rank(model, features)
        return {
            "score": -self.metric(y_pred, y_test),
            "feature_ranks": feature_ranks,
        }

    @staticmethod
    def _extract_feature_rank(
        estimator: Estimator, features: List[int]
    ) -> Dict[int, float]:
        """Extract the feature rank from the input estimator. So far it can only handle
        estimators as scikit.learn ones, so either having `the feature_importances_` or
        the `coef_` attribute."""
        if hasattr(estimator, "feature_importances_"):
            ranks = rankdata(-estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            ranks = rankdata(-np.abs(estimator.coef_[0]))
        else:
            raise ValueError("The estimator provided has no feature importances")
        return dict(zip(features, ranks))

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
            raise ValueError("Unknown type of estimator")

    def _make_metric(self, metric: Union[str, MetricFunction]):
        """Build metric function using the input `metric`. If a metric is a string
        then is interpreted as a scikit-learn metric score, such as "accuracy".
        Else, if should be a callable accepting two input arrays."""
        if isinstance(metric, str):
            return self._make_metric_from_string(metric)
        elif hasattr(metric, "__call__"):
            return metric
        else:
            raise ValueError("Input metric is not valid")

    @staticmethod
    def _make_metric_from_string(metric_string: str) -> MetricFunction:
        if metric_string == "MISS":
            return miss_score
        elif metric_string in sklearn.metrics.SCORERS:
            return sklearn.metrics.get_scorer(metric_string)._score_func
        else:
            raise ValueError("Input metric is not a valid string")


def miss_score(y_true, y_pred):
    """MISS score: number of wrong classifications preceded by - so that the higher
    this score the better the model"""
    return -(y_true != y_pred).sum()
