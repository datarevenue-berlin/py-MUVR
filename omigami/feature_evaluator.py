from typing import Union, List
from sklearn.metrics import SCORERS, get_scorer
from scipy.stats import rankdata
import numpy as np
from omigami.data_types import Estimator, MetricFunction, RandomState, NumpyArray
from omigami.data_models import FeatureEvaluationResults, FeatureRanks, TrainTestData
from omigami.estimator import ModelTrainer
from omigami.model import make_estimator


class FeatureEvaluator:
    def __init__(
        self,
        estimator: Estimator,
        metric: Union[str, MetricFunction],
        random_state: Union[int, RandomState],
    ):
        self._model_trainer = make_estimator(estimator, random_state)
        self._metric = self._make_metric(metric)
        self._random_state = random_state
        self.n_initial_features = None

    def evaluate_features(self, evaluation_data: TrainTestData, features: List[int]) -> FeatureEvaluationResults:
        X_train = evaluation_data.train_data.X
        y_train = evaluation_data.train_data.y
        estimator = self._model_trainer.train_model(X_train, y_train)

        X_test = evaluation_data.test_data.X
        y_test = evaluation_data.test_data.y
        y_pred = estimator.predict(X_test)

        score = -self._metric(y_test, y_pred)
        ranks = self._get_feature_ranks(estimator, features)
        return FeatureEvaluationResults(test_score=score, ranks=ranks, model=estimator)

    def _make_metric(self, metric: Union[str, MetricFunction]) -> MetricFunction:
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
        if metric_string in SCORERS:
            # pylint: disable=protected-access
            return get_scorer(metric_string)._score_func
        raise ValueError("Input metric is not a valid string")

    def _get_feature_ranks(
        self, estimator: Estimator, features: Union[List[int], NumpyArray]
    ) -> FeatureRanks:
        feature_importances = estimator.feature_importances
        ranks = rankdata(-feature_importances)
        return FeatureRanks(features=features, ranks=ranks, n_feats=self.n_initial_features)


def miss_score(y_true: NumpyArray, y_pred: NumpyArray):
    """MISS score: number of wrong classifications preceded by - so that the higher
    this score the better the model"""
    return -(y_true != y_pred).sum()
