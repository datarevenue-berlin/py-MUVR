import logging
from typing import Callable, Dict, List, Tuple, TypeVar, Union

import dask
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.base import BaseEstimator
from omigami.outer_looper import OuterLooper
from omigami.model_trainer import ModelTrainer
from omigami.utils import compute_number_of_features, average_scores


NumpyArray = np.ndarray
MetricFunction = Callable[[NumpyArray, NumpyArray], float]
Split = Tuple[NumpyArray, NumpyArray]
GenericEstimator = TypeVar("GenericEstimator")
Estimator = Union[BaseEstimator, GenericEstimator]


# TODO: create data classes


class FeatureSelector:
    # TODO: docstring
    RFC = "RFC"
    MIN = "min"
    MAX = "max"
    MID = "mid"

    def __init__(
        self,
        X: NumpyArray,  # TODO: move this as fit method param
        y: NumpyArray,
        n_outer: int,
        metric: Union[str, MetricFunction],
        estimator: Union[str, Estimator],
        features_dropout_rate: float = 0.05,
        robust_minimum: float = 0.05,
        n_inner: int = None,
        groups: NumpyArray = None,
        repetitions: int = 8,
        random_state: int = None,
    ):
        self.X = X
        self.y = y
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

        if groups is None:
            logging.debug("groups is not specified: i.i.d. samples assumed")
            groups = np.arange(self.X.shape[0])
        self.n_features = self.X.shape[1]
        self.groups = groups
        self._results = None
        self._selected_features = None

    def select_features(self) -> Dict[str, set]:
        """This method implement the MUVR method from:
        https://academic.oup.com/bioinformatics/article/35/6/972/5085367

        Perform recursive feature selection using nested cross validation to select
        the optimal number of features that explain the relationship between
        `self.X` and `self.y`.
        The alforithm return three sets of features:
        1. `self.MIN`: is the minimum number of feature that gives good predictive power
        2. `self.MAX`: is the maximum number of feature that gives good predictive power
        3. `self.MID`: is the set of features to build a model using a number of feature
            that is the geometric mean of the minimum and the maximum number of features

        The structure of the nested CV loops is the following:
        - Repetitions
            - Outer CV Loops
                - Iterative Feature removal
                    - Inner CV loops
        The inner loop are used to understand which feature to drop at each
        iteration removal.
        For each outer loop element, we have a curve linking the fitness to the
        number of features and average ranks for each variable.
        From the average of these curves the number of variables for each "best" model
        are extracted and the feature rank of the best models are computed.

        Averaging the resuñts and the feature importances across repetition we select
        the final set of features.

        For additional informations about the algorithm, please check the original
        paper linked above.

        Returns:
            Dict[str, set]: The 2 sets of selected features, "min", "mid", "max".
        """
        results_futures = []
        for _ in range(self.repetitions):
            model_trainer = ModelTrainer(
                X=self.X,
                y=self.y,
                groups=self.groups,
                n_inner=self.n_inner,
                n_outer=self.n_outer,
                estimator=self.estimator,
                metric=self.metric,
            )
            ol = OuterLooper(
                n_inner=self.n_inner,
                n_outer=self.n_outer,
                features_dropout_rate=self.features_dropout_rate,
                robust_minimum=self.robust_minimum,
                model_trainer=model_trainer,
            )
            repetition_futures = ol.run()
            results_futures.append(repetition_futures)
        results = dask.compute(
            results_futures,
            # scheduler="single-threaded"  #TODO: put single thread if env.DEBUG=True
        )[0]
        self._results = results
        self._selected_features = self._process_results(results)
        return self._selected_features

    def _process_results(self, results: List) -> Dict[str, set]:
        """Process the input list of outer loop results and returns the three sets
        of selected features.
        The input list is composed by outputs of `self._perform_outer_loop_cv`,
        which means that each of the self.n_repetitions elements is the result of all
        train-test cycle on outer segments fold of the data.
        """
        outer_loop_aggregation = [self._process_outer_loop(ol) for ol in results]
        return self._select_best_features(outer_loop_aggregation)

    def _process_outer_loop(self, outer_loop_results: List) -> Dict:
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
        self, outer_loop_results: List
    ) -> Dict[str, pd.DataFrame]:
        """Compute the average feature rank from a list of outputs of
        `_perform_outer_loop_cv`.
        """
        outer_test_results = [r["test_results"] for r in outer_loop_results]
        avg_feature_rank = {}
        for key in {self.MIN, self.MAX, self.MID}:
            feature_ranks = [res[key]["feature_ranks"] for res in outer_test_results]
            avg_feature_rank[key] = (
                pd.DataFrame(feature_ranks).fillna(0).mean().to_dict()
            )
        return avg_feature_rank

    def _select_best_features(self, results: List) -> Dict[str, set]:
        """Select the best features set from the outer loop aggregated results.
        The input is a list with n_repetitions elements."""
        final_feature_ranks = self._compute_final_ranks(results)
        avg_scores = [average_scores(r["scores"]) for r in results]
        n_feats = compute_number_of_features(avg_scores, self.robust_minimum)
        feature_sets = {}
        for key in (self.MIN, self.MAX, self.MID):
            feats = final_feature_ranks.sort_values(by=key).head(n_feats[key])
            feature_sets[key] = set(feats[key].index)
        return feature_sets

    def _compute_final_ranks(self, results: List) -> pd.DataFrame:
        """Average the ranks for the three sets to abaine a definitive feature rank"""
        final_ranks = {}
        for key in (self.MIN, self.MAX, self.MID):
            avg_ranks = [r["avg_feature_ranks"][key] for r in results]
            final_ranks[key] = (
                pd.DataFrame(avg_ranks).fillna(self.n_features).mean().to_dict()
            )
        return pd.DataFrame.from_dict(final_ranks).fillna(self.n_features)

    def plot_validation_curves(self) -> Axes:  # TODO: move plotting out
        """Plot validation curved using feature elimination results. The function
        will plot the relationship between finess score and number of variables
        for each outer loop iteration, their average aggregation (which gives the
        values for the single repetitions) and the average across repetitions.
        The size of the "min", "max" and "mid" sets are plotted as vertical dashed
        lines.
        """
        if self._results is None or self._selected_features is None:
            logging.warning(
                "Validation curves have not been generated. To be able to plot"
                + " call `select_features` method first"
            )
        outer_loop_aggregation = [self._process_outer_loop(ol) for ol in self._results]
        all_scores = []
        for j, res in enumerate(outer_loop_aggregation):
            for i, score in enumerate(res["scores"]):
                label = "Outer loop average" if i + j == 0 else None
                sorted_score_items = sorted(score.items())
                n_feats, score_values = zip(*sorted_score_items)
                all_scores += score_values
                plt.semilogx(n_feats, score_values, c="#deebf7", label=label)
        max_score = max(all_scores)
        min_score = min(all_scores)
        repetition_averages = []
        for i, r in enumerate(outer_loop_aggregation):
            label = "Repetition average" if i == 0 else None
            avg_scores = average_scores(r["scores"])
            sorted_score_items = sorted(avg_scores.items())
            n_feats, score_values = zip(*sorted_score_items)
            plt.semilogx(n_feats, score_values, c="#3182bd", label=label)
            repetition_averages.append(avg_scores)
        final_avg = average_scores(repetition_averages)
        sorted_score_items = sorted(final_avg.items())
        n_feats, score_values = zip(*sorted_score_items)
        plt.semilogx(n_feats, score_values, c="k", lw=3, label="Final average")
        for key in (self.MIN, self.MAX, self.MID):
            n_feats = len(self._selected_features[key])
            plt.vlines(
                n_feats,
                min_score,
                max_score,
                linestyle="--",
                colors="grey",
                lw=2,
                label=key,
            )
        plt.xlabel("# features")
        plt.ylabel("Fitness score")
        plt.grid(ls=":")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
        return plt.gca()
