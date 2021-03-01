from typing import List
import numpy as np
from scipy.stats import mode
from omigami.feature_selector import FeatureSelector
from omigami.exceptions import NotFitException
from omigami.data_structures.data_types import NumpyArray
from omigami.models import Estimator


class ConsensusModel:
    def __init__(
        self, feature_selector: FeatureSelector, feature_set_label: str, problem: str
    ):
        """Implement consensus model based on the input feature selector.
        The members of the pool are the models trained on the outer loop test folds
        during `feature_selector.fit` execution. Only estimators relative to the input
        `feature_set_label` are considered.
        The user should also provide a correct `problem` argument, to identify
        the type of prediction the consensus model must perform. `problem`
        must be based on the estimator parameter that was used to build the
        `feature_selector`. If a regressor was used, then `problem="regression"`,
        otherwise `problem="classification"`.

        Parameters
        ----------
        feature_selector : FeatureSelector
            The feature selector from which to take the outer loop models
        feature_set_label : str
            One of "min", "mid" and "max"
        problem : str
            Either "classification" or "regression" depending
            on `feature_selector.estimator`

        Raises
        ------
        NotFitException
            If the feature selector was not fit yet
        ValueError
            If `feature_set` is not one of "min", "mid" or "max"
        ValueError
            If `problem` is not one of "classification" or "regression"

        Examples
        --------
        >>> fs = FeatureSelector(n_outer=6, metric="MISS", estimator="PLSC")
        >>> fs.fit(X, y)
        >>> cm = ConsensusModel(fs, "min", "classification")
        >>> y_pred = cm.predict(additional_X)
        """
        if not feature_selector.is_fit:
            raise NotFitException("The feature selector has not been fit yet")
        if feature_set_label not in {"min", "max", "mid"}:
            raise ValueError("'feature_set' must be one of 'min', 'mid' or 'max'")
        self._models = self._extract_evaluators(feature_selector, feature_set_label)
        self._feature_sets = self._extract_feature_sets(
            feature_selector, feature_set_label
        )
        self.feature_set_label = feature_set_label
        self.n_features = feature_selector.n_features
        if problem not in ("classification", "regression"):
            raise ValueError("'problem' must be 'classification' or 'regression'")
        self.problem = problem

    def ensemble_predict(self, X: NumpyArray) -> NumpyArray:
        """Perform a prediction for the input `X` for each one of the outer loop models
        of the input feature selector.

        Parameters
        ----------
        X : NumpyArray
            Predictor variables as numpy array

        Returns
        -------
        NumpyArray
            Predicted response vectors, as matrix with as many rows as models

        Raises
        ------
        ValueError
            if input X number of columns doesn't match the feature_selector number
            of features
        """

        if X.shape[1] != self.n_features:
            raise ValueError()
        y_preds = []
        for feats, model in zip(self._feature_sets, self._models):
            pred_X = X[:, feats]
            y_pred = model.predict(pred_X)
            y_preds.append(y_pred)
        y_preds = np.vstack(y_preds)  # n-rows = n-models
        return y_preds

    def predict(self, X: NumpyArray) -> NumpyArray:
        """Permorm ensemble prediction aggegating single predictions from the
        feature_selector models. If the problem is "classification" the aggregation is
        the mode of the predicted classes, otherwise is the mean of the predicted
        response vectors.

        Parameters
        ----------
        X : NumpyArray
            Predictor variables as numpy array

        Returns
        -------
        NumpyArray
            Predicted response vector
        """
        y_preds = self.ensemble_predict(X)
        if self.problem == "classification":
            return mode(y_preds, axis=0).mode.ravel()
        return y_preds.mean(axis=0)

    # TODO: passthrough can probably remove
    @staticmethod
    def _extract_evaluators(
        feature_selector: FeatureSelector, feature_set_label: str
    ) -> List[Estimator]:
        return feature_selector.get_all_feature_models(feature_set_label)

    @staticmethod
    def _extract_feature_sets(
        feature_selector: FeatureSelector, feature_set_label: str
    ) -> List[List[int]]:
        return feature_selector.get_all_feature_sets(feature_set_label)
