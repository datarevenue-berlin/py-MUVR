import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator

from omigami.data_structures import RandomState
from omigami.models.estimator import Estimator


class ScikitLearnEstimator(Estimator):
    """Wraps scikit-learn models. This class exposes the fit and predict method of
    its input estimator. The estimator must be a scikit-learn model, such as
    RandomForestClassifier, LogisticRegression, etc.
    Additionally it can clone itself with the method `clone`. This is useful
    if the class should remain state-less.
    The input random state is helf for cloning purposes and it is used to replace the
    random state if the input estimator.
    The feature_importance exposed by this class depends on the input estimator:
    If the estimator has a `feature_importance_` attribute, than it is returned,
    otherwise the absolute values of the `coef_[0]` vector are returned. This means
    that in the exotic case of an estimator implementing both, `feature_importance_` is
    returned

    Parameters
    ----------
    estimator : BaseEstimator
        The wrapped estimator
    random_state: RandomState
        a random state instance to control reproducibility
    """

    def __init__(self, estimator: BaseEstimator, random_state: RandomState):
        self._estimator = estimator
        self._random_state = random_state
        self.set_random_state(random_state)

    @property
    def feature_importances(self):
        if hasattr(self._estimator, "feature_importances_"):
            return self._estimator.feature_importances_
        if hasattr(self._estimator, "coef_"):
            return np.abs(self._estimator.coef_[0])
        raise ValueError("The estimator provided has no feature importances")

    def set_random_state(self, random_state: RandomState):
        try:
            self._estimator.set_params(random_state=random_state)
        except ValueError:
            pass  # not all models have a random_state param (e.g. LinearRegression)

    def clone(self):
        estimator_clone = clone(self._estimator)
        return self.__class__(estimator_clone, self._random_state)

    def fit(self, X, y):
        self._estimator.fit(X, y)
        return self

    def predict(self, X):
        return self._estimator.predict(X)


class ScikitLearnPipeline(ScikitLearnEstimator):
    """Extends ScikitLearnEstimator to scikitlearn pipelines. The  feature importance
    of the pipeline is the first attribute found between `feature_importances_` or
    `abs(coef_[0])` among its various steps. This should normally correspond to the
    last one, but more unorthodox cases are supported.
    """

    @property
    def feature_importances(self):
        for _, step in self._estimator.steps:
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
