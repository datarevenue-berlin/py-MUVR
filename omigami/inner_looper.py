from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, TypeVar
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from omigami.model_trainer import TrainingTestingResult

NumpyArray = np.ndarray
GenericEstimator = TypeVar("GenericEstimator")
Estimator = Union[BaseEstimator, GenericEstimator]
Split = Tuple[NumpyArray, NumpyArray]


@dataclass
class InnerCVResult:
    train_results: List[TrainingTestingResult]
    features: List[int]

    def __len__(self):
        return len(self.train_results)

    def __getitem__(self, item_idx):
        return self.train_results[item_idx]


class InnerLooper:
    # TODO: docstring

    def __init__(self, outer_index, features_dropout_rate, model_trainer):
        self.n_inner = model_trainer.n_inner
        self.splits = [(outer_index, j) for j in range(self.n_inner)]
        assert outer_index < model_trainer.n_outer
        self.features_dropout_rate = features_dropout_rate
        self.model_trainer = model_trainer
        self.n_features = self.model_trainer.n_features

    def run(self) -> Dict[Tuple[int], List]:
        """Perform inner loop cross validation using all the inner loop splits derived
        from the outer split i. The inner loop performs iteratve variable removal.
        At each step a fraction `self.feature_dropout_rate`  of the features is removed.
        To choose the features to remove a CV based on the `self.n_inner`splits is
        performed. It return a dectionary containing the results of every train-test
        CV at each step of the iterative variable removal. Each key corresponds to the
        set of features used at that removal iteration.

        Args:
            i (int): outer loop index

        Returns:
            Dict[Tuple[int], List]: variable removal train-test results.
        """
        final_results = {}
        features = list(range(self.n_features))
        while len(features) > 1:
            train_results = [self.model_trainer.run(s, features) for s in self.splits]
            inner_cv_res = InnerCVResult(
                train_results=train_results, features=features,
            )
            final_results[tuple(features)] = inner_cv_res
            features = self._keep_best_features(inner_cv_res)
        return final_results

    def _keep_best_features(self, inner_cv_results: InnerCVResult) -> List[int]:
        """Keep the best features based on their average rank"""
        feature_ranks = [r["feature_ranks"] for r in inner_cv_results.train_results]
        avg_ranks = pd.DataFrame(feature_ranks).fillna(self.n_features).mean().to_dict()
        for f in inner_cv_results.features:
            if f not in avg_ranks:
                avg_ranks[f] = self.n_features
        sorted_averages = sorted(avg_ranks.items(), key=lambda x: x[1])
        n_feats = len(inner_cv_results.features)
        n_features_to_drop = round(self.features_dropout_rate * n_feats)
        if not n_features_to_drop:
            n_features_to_drop = 1
        sorted_averages = sorted_averages[:-n_features_to_drop]
        return [feature for feature, _ in sorted_averages]
