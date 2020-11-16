from dataclasses import dataclass
from typing import Dict, List
import dask
from omigami.inner_looper import InnerLooper, InnerLoopResults
from omigami.model_trainer import TrainingTestingResult
import omigami.utils as utils


@dataclass
class OuterLoopModelTrainResults:
    MIN: TrainingTestingResult
    MAX: TrainingTestingResult
    MID: TrainingTestingResult


@dataclass
class OuterLoopResults:
    test_results: OuterLoopModelTrainResults
    scores: dict

    def __getitem__(self, key):
        return self.__getattribute__(key)


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
    def _perform_outer_loop_cv(self, outer_index: int) -> OuterLoopResults:
        """Perform an outer loop cross validation on all the splits linked to the
        outer fold `i`. It returns a dictionary containing the fitness score of the
        loop and the results of the prediction using the selected best features on
        the outer test fold

        Args:
            outer_index (int): index of the loop

        Returns:
            Dict: Results of the outer loop training and testing
        """

        inner_looper = InnerLooper(
            outer_index=outer_index,
            features_dropout_rate=self.features_dropout_rate,
            model_trainer=self.model_trainer,
        )
        inner_loop_results = inner_looper.run()
        res = self._select_best_features_and_score(inner_loop_results)

        outer_split_id = (outer_index,)
        outer_test_results = OuterLoopModelTrainResults(
            MIN=self.model_trainer.run(outer_split_id, res[self.MIN]),
            MID=self.model_trainer.run(outer_split_id, res[self.MID]),
            MAX=self.model_trainer.run(outer_split_id, res[self.MAX]),
        )

        return OuterLoopResults(
            test_results=outer_test_results, scores=inner_loop_results.score
        )

    def _select_best_features_and_score(
        self, inner_loop_results: InnerLoopResults
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

        n_feats = utils.compute_number_of_features(
            [inner_loop_results.score], self.robust_minimum
        )
        max_feats = n_feats[self.MAX]
        min_feats = n_feats[self.MIN]
        mid_feats = n_feats[self.MID]
        mid_feats = inner_loop_results.get_closest_number_of_features(mid_feats)
        return {
            "min": inner_loop_results.get_features_from_their_number(min_feats),
            "max": inner_loop_results.get_features_from_their_number(max_feats),
            "mid": inner_loop_results.get_features_from_their_number(mid_feats),
            "score": inner_loop_results.score,
        }
