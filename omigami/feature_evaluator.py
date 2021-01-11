from typing import Union, List
from scipy.stats import rankdata
from omigami.data_structures import (
    MetricFunction,
    RandomState,
    NumpyArray,
    FeatureEvaluationResults,
    FeatureRanks,
    TrainTestData,
)
from omigami.models.metrics import make_metric
from omigami.models.estimator import make_estimator, Estimator


class FeatureEvaluator:
    def __init__(
        self,
        estimator: Estimator,
        metric: Union[str, MetricFunction],
        random_state: Union[int, RandomState],
    ):
        self._estimator = make_estimator(estimator, random_state)
        self._metric = make_metric(metric)
        self._n_initial_features = 0

    def set_n_initial_features(self, n_initial_features: int):
        self._n_initial_features = n_initial_features

    def evaluate_features(
        self, evaluation_data: TrainTestData, features: List[int]
    ) -> FeatureEvaluationResults:
        X_train = evaluation_data.train_data.X
        y_train = evaluation_data.train_data.y
        X_test = evaluation_data.test_data.X
        y_test = evaluation_data.test_data.y

        estimator = self._estimator.clone().fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        score = -self._metric(y_test, y_pred)
        ranks = self._get_feature_ranks(estimator, features)
        return FeatureEvaluationResults(test_score=score, ranks=ranks, model=estimator)

    def _get_feature_ranks(
        self, estimator: Estimator, features: Union[List[int], NumpyArray]
    ) -> FeatureRanks:
        if self._n_initial_features == 0:
            raise ValueError("Call set_n_initial_features first")
        feature_importances = estimator.feature_importances
        ranks = rankdata(-feature_importances)
        return FeatureRanks(
            features=features, ranks=ranks, n_feats=self._n_initial_features
        )
