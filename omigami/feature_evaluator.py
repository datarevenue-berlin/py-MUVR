from typing import Union, List
from scipy.stats import rankdata
import numpy as np
from omigami.data_types import Estimator, MetricFunction, RandomState, NumpyArray
from omigami.data_models import FeatureEvaluationResults, FeatureRanks, TrainTestData
from omigami.metrics import make_metric
from omigami.model import make_estimator


class FeatureEvaluator:
    def __init__(
        self,
        estimator: Estimator,
        metric: Union[str, MetricFunction],
        random_state: Union[int, RandomState],
    ):
        self._model_trainer = make_estimator(estimator, random_state)
        self._metric = make_metric(metric)
        self._random_state = random_state
        self.n_initial_features = None

    def evaluate_features(
        self, evaluation_data: TrainTestData, features: List[int]
    ) -> FeatureEvaluationResults:
        X_train = evaluation_data.train_data.X
        y_train = evaluation_data.train_data.y
        estimator = self._model_trainer.train_model(X_train, y_train)

        X_test = evaluation_data.test_data.X
        y_test = evaluation_data.test_data.y
        y_pred = estimator.predict(X_test)

        score = -self._metric(y_test, y_pred)
        ranks = self._get_feature_ranks(estimator, features)
        return FeatureEvaluationResults(test_score=score, ranks=ranks, model=estimator)

    def _get_feature_ranks(
        self, estimator: Estimator, features: Union[List[int], NumpyArray]
    ) -> FeatureRanks:
        feature_importances = estimator.feature_importances
        ranks = rankdata(-feature_importances)
        return FeatureRanks(
            features=features, ranks=ranks, n_feats=self.n_initial_features
        )
