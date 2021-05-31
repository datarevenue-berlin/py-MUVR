from typing import List

import numpy as np
from scipy.stats import mode
from sklearn.base import is_classifier

from py_muvr.feature_selector import FeatureSelector
from py_muvr.data_structures.data_types import NumpyArray
from py_muvr.models import Estimator


class ConsensusModel:
    def __init__(self, feature_selector: FeatureSelector, feature_set_label: str):
        """Implement consensus model based on the input feature selector.
        The members of the pool are the models trained on the outer loop test folds
        during `feature_selector.fit` execution. Only estimators relative to the input
        `feature_set_label` are considered.
        The user should also provide a correct `problem` argument, to identify
        the type of prediction the consensus model must perform. `problem`
        must be based on the estimator parameter that was used to build the
        `feature_selector`.

        Parameters
        ----------
        feature_selector : FeatureSelector
            The feature selector from which to take the outer loop models
        feature_set_label : str
            One of "min", "mid" and "max"

        Raises
        ------
        NotFitException
            If the feature selector was not fit yet
        ValueError
            If `feature_set` is not one of "min", "mid" or "max"

        Examples
        --------
        >>> fs = FeatureSelector(n_outer=6, metric="MISS", estimator="PLSC")
        >>> fs.fit(X, y)
        >>> cm = ConsensusModel(fs, "min", "classification")
        >>> y_pred = cm.predict(additional_X)
        """
        if feature_set_label not in {"min", "max", "mid"}:
            raise ValueError("'feature_set' must be one of 'min', 'mid' or 'max'")
        self._models = self._get_all_models(feature_selector, feature_set_label)
        self._feature_sets = self._get_all_feature_sets(
            feature_selector, feature_set_label
        )
        self.feature_set_label = feature_set_label

    @staticmethod
    def _get_all_feature_sets(selector: FeatureSelector, label: str) -> List[List[int]]:
        attr_name = label + "_eval"
        flat_results = [r for repetition in selector.raw_results for r in repetition]
        ranks = [getattr(result, attr_name).ranks for result in flat_results]
        feature_sets = [list(r.get_data().keys()) for r in ranks]
        return feature_sets

    @staticmethod
    def _get_all_models(selector: FeatureSelector, label: str) -> List[Estimator]:
        flat_results = [r for repetition in selector.raw_results for r in repetition]
        attr_name = label + "_eval"
        return [getattr(result, attr_name).model for result in flat_results]

    def _ensemble_predict(self, X: NumpyArray) -> NumpyArray:
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
        y_preds = []
        for feats, model in zip(self._feature_sets, self._models):
            pred_X = X[:, feats]
            y_pred = model.predict(pred_X)
            y_preds.append(y_pred)
        y_preds = np.vstack(y_preds)  # n-rows = n-models
        return y_preds

    def predict(self, X: NumpyArray) -> NumpyArray:
        """Perform ensemble prediction aggregating single predictions from the
        feature_selector models. If the problem is a classification the aggregation is
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
        y_preds = self._ensemble_predict(X)
        if is_classifier(self._models[0]):
            return mode(y_preds, axis=0)[0].ravel()
        return y_preds.mean(axis=0)
