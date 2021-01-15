from typing import List
import numpy as np
from scipy.stats import mode
from omigami.feature_selector import FeatureSelector
from omigami.exceptions import NotFitException
from omigami.data_structures.data_types import NumpyArray
from omigami.models import Estimator


class ConsensusModel:
    def __init__(self, feature_selector: FeatureSelector, model: str, problem: str):
        if not feature_selector.is_fit:
            raise NotFitException("The feature selector has not been fit yet")
        if model not in {"min", "max", "mid"}:
            raise ValueError("'model' must be one of 'min', 'mid' or 'max'")
        self._models = self._extract_evaluators(feature_selector, model)
        self._feature_sets = self._extract_feature_sets(feature_selector, model)
        self.model = model
        self.n_features = feature_selector.n_features
        if problem not in ("classification", "regression"):
            raise ValueError("'problem' must be 'classification' or 'regression'")
        self.problem = problem

    def raw_predict(self, X: NumpyArray) -> NumpyArray:
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
        y_preds = self.raw_predict(X)
        if self.problem == "classification":
            return mode(y_preds, axis=0).mode.ravel()
        return y_preds.mean(axis=0)

    @staticmethod
    def _extract_evaluators(
        feature_selector: FeatureSelector, model: str
    ) -> List[List[int]]:
        return feature_selector.post_processor.get_all_feature_models(
            feature_selector.results, model
        )

    @staticmethod
    def _extract_feature_sets(
        feature_selector: FeatureSelector, model: str
    ) -> List[Estimator]:
        return feature_selector.post_processor.get_all_feature_sets(
            feature_selector.results, model
        )
