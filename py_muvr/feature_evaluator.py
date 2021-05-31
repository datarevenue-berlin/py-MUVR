from typing import Union, List
from scipy.stats import rankdata
from py_muvr.data_structures import (
    MetricFunction,
    RandomState,
    NumpyArray,
    FeatureEvaluationResults,
    FeatureRanks,
    TrainTestData,
)
from py_muvr.models.metrics import make_metric
from py_muvr.models import make_estimator, Estimator


class FeatureEvaluator:
    """
    This class is used to evaluate a set of features given an estimator, a scoring
    metric and a random state.

    Parameters
    ----------
    estimator: Estimator
        Model used to evaluate feature sets
    metric: Union[str, MetricFunction]
        Metric used to measure the performance of the model using the feature set.
    random_state: Union[int, RandomState]
        A random state instance to control reproducibility
    """

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
        """
        Saves the initial number of features to be used as a reference to compute the
        average of all feature ranks from different feature subsets.

        Parameters
        ----------
        n_initial_features: int
            The initial number of features to use as a reference for feature ranks.

        """
        self._n_initial_features = n_initial_features

    def evaluate_features(
        self, evaluation_data: TrainTestData, features: List[int]
    ) -> FeatureEvaluationResults:
        """
        This method evaluates a feature set on an input data. This is done by training
        the estimator on a train set, predicting for a validation set and computing the
        score on the validation set based on the chosen metric.

        Parameters
        ----------
        evaluation_data: TrainTestData
            Dataset object containing train/test X and y
        features:
            Feature set being used on the evaluation data

        Returns
        -------
        FeatureEvaluationResults:
            The results of the evaluation. It consists of:
            - features: the feature set
            - ranks: the ranking of the features
            - model: the estimator used on evaluation

        """
        if self._n_initial_features == 0:
            raise ValueError("Call set_n_initial_features first")

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
        feature_importances = estimator.feature_importances
        ranks = rankdata(-feature_importances)
        return FeatureRanks(
            features=features, ranks=ranks, n_feats=self._n_initial_features
        )
