from dataclasses import dataclass
from typing import Dict, List

import dask
from sklearn.model_selection import GroupKFold

from omigami.model_trainer import TrainingTestingResult, ModelTrainer
from omigami.recursive_feature_eliminator import (
    RecursiveFeatureEliminator,
    RecursiveFeatureEliminatorResults,
)
from omigami.utils import MIN, MAX, MID, NumpyArray, compute_number_of_features


@dataclass
class OuterLoopModelTrainResults:
    MIN: TrainingTestingResult
    MAX: TrainingTestingResult
    MID: TrainingTestingResult

    _attribute_map = {
        MIN: "MIN",
        MID: "MID",
        MAX: "MAX",
    }

    def __getitem__(self, key):
        attribute = self._attribute_map[key]
        return self.__getattribute__(attribute)


# TODO: why not a single dataclass?
@dataclass
class OuterLoopResults:
    test_results: OuterLoopModelTrainResults
    scores: dict

    def __getitem__(self, key):
        return self.__getattribute__(key)


class OuterLooper:
    """Class that performs the outer loop CV feature selection.

    Args:
        features_dropout_rate (float): fraction of features to drop at each elimination
            step
        robust_minimum (float): maximum normalized-score value to be considered when
            computing the selected features
        model_trainer (ModelTrainer): object that trains the model over the splits

    """

    def __init__(
        self,
        robust_minimum: float,
        n_outer: float,
        groups: NumpyArray,
    ):
        self.n_outer = n_outer
        self.robust_minimum = robust_minimum
        self.groups = groups

    def run(
        self,
        X: NumpyArray,
        y: NumpyArray,
        model_trainer: ModelTrainer,
        rfe: RecursiveFeatureEliminator,
    ) -> List[OuterLoopResults]:

        outer_splits = self._make_outer_splits(X, y)
        outer_loop_results = []

        for outer_index, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
            outer_fold_results = self._run_outer_fold(
                X, y, model_trainer, rfe, outer_train_idx, outer_test_idx
            )
            outer_loop_results.append(outer_fold_results)

        return outer_loop_results

    @dask.delayed
    def _run_outer_fold(
        self,
        X: NumpyArray,
        y: NumpyArray,
        model_trainer: ModelTrainer,
        rfe: RecursiveFeatureEliminator,
        outer_train_ix: NumpyArray,
        outer_test_ix: NumpyArray,
    ):
        X_outer, y_outer = X[outer_train_ix], y[outer_train_ix]
        feature_elim_results = rfe.run(
            X_outer, y_outer, model_trainer, self.groups[outer_train_ix]
        )

        res = self._select_best_features_and_score(feature_elim_results)

        outer_test_results = OuterLoopModelTrainResults(
            MIN=model_trainer.evaluate_features(
                X, y, outer_train_ix, outer_test_ix, res[MIN]
            ),
            MID=model_trainer.evaluate_features(
                X, y, outer_train_ix, outer_test_ix, res[MID]
            ),
            MAX=model_trainer.evaluate_features(
                X, y, outer_train_ix, outer_test_ix, res[MAX]
            ),
        )
        outer_fold_results = OuterLoopResults(
            test_results=outer_test_results, scores=feature_elim_results.score
        )

        return outer_fold_results

    def _select_best_features_and_score(
        self, rfe_results: RecursiveFeatureEliminatorResults
    ) -> Dict:
        """Select the best features analysing the train results across the feature
        removal cycle. Each step of the cycle is an entry in the input dictionary

            {features_used: [res_fold_1, ..., res_fold_n_inner]}

        First, an overall score is obtained summing all the scores (for consistency
        with the original R package, MUVR. Averaging would not change the result.)
        Next, the appropriate number of features is extracted from this condensed score.
        Then, the feature set is established using the steps in the feature elimination
        having the number of features computed in the previous step.

        The score and the features are returned as dict.

        Args:
            outer_train_results (Dict[Tuple[int], List]): Result of the feature
            elimination

        Returns:
            Dict: The best feature for each of the three sets and the condensed score
        """

        n_feats = compute_number_of_features([rfe_results.score], self.robust_minimum)
        max_feats = n_feats[MAX]
        min_feats = n_feats[MIN]
        mid_feats = n_feats[MID]
        mid_feats = rfe_results.get_closest_number_of_features(mid_feats)
        return {
            MIN: rfe_results.get_features_from_their_number(min_feats),
            MAX: rfe_results.get_features_from_their_number(max_feats),
            MID: rfe_results.get_features_from_their_number(mid_feats),
            "score": rfe_results.score,
        }

    def _make_outer_splits(self, X: NumpyArray, y: NumpyArray):
        outer_splitter = GroupKFold(self.n_outer)
        outer_splits = outer_splitter.split(X, y, self.groups)

        return outer_splits
