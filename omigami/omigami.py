import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TypeVar, Union

import dask
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from omigami.outer_looper import OuterLooper, OuterLoopResults
from omigami.model_trainer import ModelTrainer, FeatureRanks
from omigami.utils import compute_number_of_features, average_scores, MIN, MAX, MID

NumpyArray = np.ndarray
MetricFunction = Callable[[NumpyArray, NumpyArray], float]
Split = Tuple[NumpyArray, NumpyArray]
GenericEstimator = TypeVar("GenericEstimator")
Estimator = Union[BaseEstimator, GenericEstimator]


@dataclass
class SelectedFeatures:
    MIN: set
    MAX: set
    MID: set

    _attribute_map = {
        MIN: "MIN",
        MID: "MID",
        MAX: "MAX",
    }

    def __getitem__(self, key: str):
        attribute = self._attribute_map[key]
        return self.__getattribute__(attribute)


class FeatureSelector:
    """Feature selection based on double cross validation and iterative feature
    elimination.

    This class is based on the feature selection algorithm proposed in
    "Variable selection and validation in multivariate modelling", Shi L. et al.,
    Bioinformatics 2019
    https://academic.oup.com/bioinformatics/article/35/6/972/5085367

    Args:
        n_outer (int): number of outer CV folds
        metric (str, callable): metric to be used to assess estimator goodness
        estimator (str, BaseEstimator): estimator to be used for feature elimination
        features_dropout_rate (float): fraction of features to drop at each elimination
            step
        robust_minimum (float): maximum normalized-score value to be considered when
            computing the selected features
        n_inner (int): number of inner CV folds (default: n_outer - 1)
        repetitions (int): number of repetitions of the double CV loops (default: 8)
        random_state (int): pass an int for reproducible output (default: None)

    Examples:
        >>> X, y = load_some_dataset()
        >>> feature_selector = FeatureSelector(10, "MISS", "RFC")
        >>> selected_feats = feature_selector.fit(X, y)
        >>> selected_feats.MIN
        {0, 1, 2}

    """

    def __init__(
        self,
        n_outer: int,
        metric: Union[str, MetricFunction],
        estimator: Union[str, Estimator],
        features_dropout_rate: float = 0.05,
        robust_minimum: float = 0.05,
        n_inner: int = None,
        repetitions: int = 8,
        random_state: int = None,
    ):
        self.is_fit = False
        self.random_state = random_state
        self.n_outer = n_outer
        self.metric = metric
        self.estimator = estimator
        self.features_dropout_rate = features_dropout_rate
        self.robust_minimum = robust_minimum
        self.repetitions = repetitions

        if not n_inner:
            logging.debug("n_inner is not specified, setting it to n_outer - 1")
            n_inner = n_outer - 1
        self.n_inner = n_inner

        self.n_features = None
        self.selected_features = None
        self.outer_loop_aggregation = None
        self._results = None

    def fit(
        self, X: NumpyArray, y: NumpyArray, groups: NumpyArray = None
    ) -> SelectedFeatures:
        """This method implement the MUVR method from:
        https://academic.oup.com/bioinformatics/article/35/6/972/5085367 and
        https://gitlab.com/CarlBrunius/MUVR/-/tree/master/R

        Perform recursive feature selection using nested cross validation to select
        the optimal number of features explaining the relationship between `X` and `y`.
        The alforithm return three sets of features, stored in SelectedFeatures:
        1. `MIN`: is the minimum number of feature that gives good predictive power
        2. `MAX`: is the maximum number of feature that gives good predictive power
        3. `MID`: is the set of features to build a model using a number of feature
            that is the geometric mean of the minimum and the maximum number of features

        The structure of the nested CV loops is the following:
        - Repetitions
            - Outer CV Loops
                - Iterative Feature removal
                    - Inner CV loops
        The inner loop are used to understand which feature to drop at each
        iteration removal.
        For each outer loop element, we have a score curve linking the fitness to the
        number of features, and average ranks for each variable.
        From the average of these curves, the number of variables for each "best" model
        (MIN, MID and MAX) are extracted and the feature rank of the best models
        are computed.

        Averaging the results and the feature importances across repetitions,  we select
        the final set of features.

        For additional informations about the algorithm, please check the original
        paper linked above.

        Returns:
            SelectedFeatures: The 3 sets of selected features, MIN, MID and MAX.
        """

        if groups is None:
            logging.debug("groups is not specified: i.i.d. samples assumed")
            groups = np.arange(X.shape[0])

        self.n_features = X.shape[1]

        results_futures = []
        for repetition_idx in range(self.repetitions):
            model_trainer = self._build_model_trainer(repetition_idx, X, y, groups)
            outer_looper = self._build_outer_looper(model_trainer)
            repetition_futures = outer_looper.run()
            results_futures.append(repetition_futures)

        results = dask.compute(
            results_futures,
            # scheduler="single-threaded"  #TODO: put single thread if env.DEBUG=True
        )[0]
        self._results = results
        self.selected_features = self._process_results(results)
        self.is_fit = True
        return self.selected_features

    def _build_model_trainer(
        self, repetition_idx: int, X: NumpyArray, y: NumpyArray, groups: NumpyArray
    ) -> ModelTrainer:
        random_state = (
            self.random_state + repetition_idx
            if self.random_state is not None
            else None
        )
        return ModelTrainer(
            X=X,
            y=y,
            groups=groups,
            n_inner=self.n_inner,
            n_outer=self.n_outer,
            estimator=self.estimator,
            metric=self.metric,
            random_state=random_state,
        )

    def _build_outer_looper(self, model_trainer: ModelTrainer) -> OuterLooper:
        return OuterLooper(
            features_dropout_rate=self.features_dropout_rate,
            robust_minimum=self.robust_minimum,
            model_trainer=model_trainer,
        )

    def _process_results(
        self, results: List[List[OuterLoopResults]]
    ) -> SelectedFeatures:
        """Process the input list of outer loop results and returns the three sets
        of selected features.
        The input list is composed by outputs of `self._perform_outer_loop_cv`,
        which means that each of the self.n_repetitions elements is the result of all
        train-test cycle on outer segments fold of the data.
        """
        outer_loop_aggregation = [self._process_outer_loop(ol) for ol in results]
        self.outer_loop_aggregation = outer_loop_aggregation
        return self._select_best_features(outer_loop_aggregation)

    def _process_outer_loop(self, outer_loop_results: List[OuterLoopResults]) -> Dict:
        """Process the self.n_outer elements of the input list to extract the condensed
        resuñts for the repetition. It return a dictionary containing
        1. the average rank of the three sets of features
        2. the length of the three sets of feature
        3. The average value of the fitness score vs number of variables across the
           n_outer elements
        """
        avg_feature_rank = self._compute_avg_feature_rank(outer_loop_results)
        scores = [r["scores"] for r in outer_loop_results]
        n_feats = compute_number_of_features(scores, self.robust_minimum)
        return {
            "avg_feature_ranks": avg_feature_rank,
            "scores": scores,
            "n_feats": n_feats,
        }

    def _compute_avg_feature_rank(
        self, outer_loop_results: List[OuterLoopResults]
    ) -> Dict[str, pd.DataFrame]:
        """Compute the average feature rank from a list of outputs of
        `_perform_outer_loop_cv`.
        """
        outer_test_results = [r["test_results"] for r in outer_loop_results]
        avg_feature_rank = {}
        for key in {MIN, MAX, MID}:
            feature_ranks = [
                res[key].feature_ranks.to_dict() for res in outer_test_results
            ]
            avg_feature_rank[key] = (
                pd.DataFrame(feature_ranks).fillna(self.n_features).mean().to_dict()
            )
        return avg_feature_rank

    def _select_best_features(self, results: List[Dict]) -> SelectedFeatures:
        """Select the best features set from the outer loop aggregated results.
        The input is a list with n_repetitions elements."""
        final_feature_ranks = self._compute_final_ranks(results)
        avg_scores = [average_scores(r["scores"]) for r in results]
        n_feats = compute_number_of_features(avg_scores, self.robust_minimum)

        feature_sets = {}
        for key in (MIN, MAX, MID):
            feats = final_feature_ranks.sort_values(by=key).head(n_feats[key])
            feature_sets[key] = set(feats[key].index)

        return SelectedFeatures(
            MID=feature_sets[MID], MIN=feature_sets[MIN], MAX=feature_sets[MAX],
        )

    def _compute_final_ranks(self, results: List[Dict]) -> pd.DataFrame:
        """Average the ranks for the three sets to abaine a definitive feature rank"""
        final_ranks = {}
        for key in (MIN, MAX, MID):
            avg_ranks = [r["avg_feature_ranks"][key] for r in results]
            final_ranks[key] = (
                pd.DataFrame(avg_ranks).fillna(self.n_features).mean().to_dict()
            )
        return pd.DataFrame.from_dict(final_ranks).fillna(self.n_features)
