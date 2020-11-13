from typing import Dict, List, Tuple
import dask
import numpy as np
from omigami.inner_looper import InnerLooper
import omigami.utils as utils


class OuterLooper:
    # TODO: docstring

    MIN = utils.MIN
    MAX = utils.MAX
    MID = utils.MID

    def __init__(
        self, n_inner, n_outer, features_dropout_rate, robust_minimum, model_trainer,
    ):
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.features_dropout_rate = features_dropout_rate
        self.robust_minimum = robust_minimum
        self.model_trainer = model_trainer

    def run(self) -> List:
        return [self._perform_outer_loop_cv(i) for i in range(self.n_outer)]

    @dask.delayed  # TODO: keep it generalizable
    def _perform_outer_loop_cv(self, outer_index: int) -> Dict:
        """Perform an outer loop cross validation on all the splits linked to the
        outer fold `i`. It returns a dictionary containing the fitness score of the
        loop and the results of the prediction using the selected best features on
        the outer test fold

        Args:
            outer_index (int): index of the loop

        Returns:
            Dict: Results of the outer loop training and testing
        """

        il = InnerLooper(
            outer_index=outer_index,
            features_dropout_rate=self.features_dropout_rate,
            model_trainer=self.model_trainer,
        )
        outer_train_results = il.run()
        res = self._select_best_features_and_score(outer_train_results)
        scores = res.pop("score")

        outer_test_results = {}
        for key in {self.MIN, self.MID, self.MAX}:
            features = res[key]
            outer_test_results[key] = self.model_trainer.run((outer_index,), features)

        return {
            "test_results": outer_test_results,
            "scores": scores,
        }

    def _select_best_features_and_score(
        self, outer_train_results: Dict[Tuple[int], List]
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
        features_kept = {}
        score = {}
        for features, res in outer_train_results.items():
            n_feats = len(features)
            features_kept[n_feats] = features
            score[n_feats] = np.sum([r["score"] for r in res])

        n_feats = utils.compute_number_of_features([score], self.robust_minimum)
        max_feats = n_feats[self.MAX]
        min_feats = n_feats[self.MIN]
        mid_feats = n_feats[self.MID]
        mid_feats = min(score.keys(), key=lambda x: abs(x - mid_feats))
        return {
            "min": features_kept[min_feats],
            "max": features_kept[max_feats],
            "mid": features_kept[mid_feats],
            "score": score,
        }
