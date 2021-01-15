from __future__ import annotations

import logging
from concurrent.futures import Executor, Future
from typing import Union, List, Dict, Tuple

import numpy as np
import progressbar
from numpy.random import RandomState

from omigami.data_structures import (
    InputDataset,
    SelectedFeatures,
    OuterLoopResults,
    InnerLoopResults,
    Split,
    FeatureEvaluationResults,
    MetricFunction,
    InputEstimator,
    NumpyArray,
    FeatureSelectionResults,
)
from omigami.feature_evaluator import FeatureEvaluator
from omigami.data_splitter import DataSplitter
from omigami.post_processor import PostProcessor
from omigami.utils import get_best_n_features, average_ranks
from omigami.exceptions import NotFitException
from omigami.sync_executor import SyncExecutor


Repetition = List[Union[OuterLoopResults, Future]]
log = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selection based on double cross validation and iterative feature
    elimination.
    This class is based on the feature selection algorithm proposed in
    "Variable selection and validation in multivariate modelling", Shi L. et al.,
    Bioinformatics 2019
    https://academic.oup.com/bioinformatics/article/35/6/972/5085367

    Perform recursive feature selection using nested cross validation to select
    the optimal number of features explaining the relationship between `X` and `y`.
    The algorithm outputs three sets of features, that can be accessed via
    self.get_selected_features.

    1. `min`: is the minimum number of feature that gives good predictive power
    2. `max`: is the maximum number of feature that gives good predictive power
    3. `mid`: is the set of features to build a model using a number of feature
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

    Averaging the results and the feature importances across repetitions, we select
    the final set of features.

    The actual feature selection is performed by the `fit` method which implements the
    algorithm described in the original paper and developed in
    https://gitlab.com/CarlBrunius/MUVR/-/tree/master/R

    For additional informations about the algorithm, please check the original
    paper linked above.

    Parameters
    ----------
    n_outer: int
        number of outer CV folds
    metric: Union[str, MetricFunction]
        metric to be used to assess estimator goodness
    estimator: Union[str, InputEstimator]
        estimator to be used for feature elimination
    features_dropout_rate: float
        fraction of features to drop at each elimination step
    robust_minimum: float
        maximum normalized-score value to be considered when computing the `min` and
        `max` selected features
    n_inner: int
        number of inner CV folds, by default n_outer - 1
    n_repetitions: int
        number of repetitions of the double CV loops, by default 8
    random_state: int
        pass an int for reproducible output, by default None
    """

    def __init__(
        self,
        n_outer: int,
        metric: Union[str, MetricFunction],
        estimator: Union[str, InputEstimator],
        features_dropout_rate: float = 0.05,
        robust_minimum: float = 0.05,
        n_inner: int = None,
        n_repetitions: int = 8,
        random_state: int = None,
    ):
        self.is_fit = False
        self._n_features = None
        self.random_state = None if random_state is None else RandomState(random_state)
        self.n_outer = n_outer
        self.keep_fraction = 1 - features_dropout_rate
        self.n_repetitions = n_repetitions
        self.feature_evaluator = FeatureEvaluator(estimator, metric, random_state)

        if not n_inner:
            log.info("Parameter n_inner is not specified, setting it to n_outer - 1")
            n_inner = n_outer - 1
        self.n_inner = n_inner

        self._selected_features = None
        self.outer_loop_aggregation = None
        self.results = None
        self._minimum_features = 1
        self.post_processor = PostProcessor(robust_minimum)

    def fit(
        self,
        X: NumpyArray,
        y: NumpyArray,
        groups: NumpyArray = None,
        executor: Executor = None,
    ) -> FeatureSelector:
        """
        Implements the double CV feature selection algorithm. The method returns
        the same FeatureSelector. If the samples are correlated, the `group` vector
        can be used to encode arbitrary domain specific stratifications of the
        samples as integers (e.g. patient_id, year of collection, etc.). If group
        is not provided the samples are assumed to be i. i. d. variables.
        To parallelize the CV repetition, an `executor` can be provided to split
        the computation across processes or cluster nodes. So far, `loky` (joblib),
        `dask`, and `concurrent` Executors are tested.

        Parameters
        ----------
        X : NumpyArray
            Predictor variables as numpy array
        y : NumpyArray
            Response vector (Dependent variable).
        groups : NumpyArray, optional
            Group labels for the samples used while splitting the dataset
            into train/test set, by default None
        executor : Executor, optional
            executor instance for parallel computing, by default None

        Returns
        -------
        FeatureSelector
            the fit feature selector
        """

        if executor is None:
            executor = SyncExecutor()

        size, n_features = X.shape
        groups = self._get_groups(groups, size)
        input_data = InputDataset(X=X, y=y, groups=groups)
        self.feature_evaluator.set_n_initial_features(n_features)

        log.info(
            f"Running {self.n_repetitions} repetitions and"
            f" {self.n_outer} outer loops using "
            f"executor {executor.__class__.__name__}."
        )

        repetition_results = []

        log.info("Scheduling tasks...")
        with progressbar.ProgressBar(max_value=self.n_repetitions * self.n_outer) as b:
            progress = 0
            b.update(progress)
            for _ in range(self.n_repetitions):
                data_splitter = DataSplitter(
                    self.n_outer,
                    self.n_inner,
                    input_data,
                    self.random_state,
                )

                outer_loop_results = []
                for outer_split in data_splitter.iter_outer_splits():
                    outer_loop_result = self._deferred_run_outer_loop(
                        input_data,
                        outer_split,
                        executor=executor,
                        data_splitter=data_splitter,
                    )
                    outer_loop_results.append(outer_loop_result)
                    progress += 1
                    b.update(progress)

                repetition_results.append(outer_loop_results)

        self._selected_features = self._select_best_features(repetition_results)
        log.info("Finished feature selection.")
        self._n_features = input_data.n_features
        self.is_fit = True
        return self

    @staticmethod
    def _get_groups(groups: NumpyArray, size: int) -> NumpyArray:
        if groups is None:
            log.info("Groups parameter is not specified: independent samples assumed")
            groups = np.arange(size)
        return groups

    def _deferred_run_outer_loop(
        self,
        input_data: InputDataset,
        outer_split: Split,
        data_splitter: DataSplitter,
        executor: Executor,
    ) -> Union[Future, OuterLoopResults]:
        if executor is None:
            return self._run_outer_loop(input_data, outer_split, data_splitter)
        return executor.submit(
            self._run_outer_loop, input_data, outer_split, data_splitter
        )

    def _run_outer_loop(
        self,
        input_data: InputDataset,
        outer_split: Split,
        data_splitter: DataSplitter,
    ) -> OuterLoopResults:

        feature_elimination_results = {}
        feature_set = list(range(input_data.n_features))

        while len(feature_set) >= self._minimum_features:
            inner_results = []

            for inner_split in data_splitter.iter_inner_splits(outer_split):
                inner_loop_data = data_splitter.split_data(
                    input_data, inner_split, feature_set
                )

                feature_evaluation_results = self.feature_evaluator.evaluate_features(
                    inner_loop_data, feature_set
                )

                inner_results.append(feature_evaluation_results)

            feature_elimination_results[tuple(feature_set)] = inner_results
            feature_set = self._remove_features(feature_set, inner_results)

        outer_loop_results = self._create_outer_loop_results(
            feature_elimination_results, input_data, outer_split, data_splitter
        )

        return outer_loop_results

    def _remove_features(
        self, features: List[int], results: InnerLoopResults
    ) -> List[int]:
        features_to_keep = int(np.floor(len(features) * self.keep_fraction))
        features = self._select_n_best(results, features_to_keep)
        return features

    @staticmethod
    def _select_n_best(inner_loop_result: InnerLoopResults, keep_n: int) -> List[int]:
        if keep_n < 1:
            return []
        ranks = [r.ranks for r in inner_loop_result]
        avg_ranks = average_ranks(ranks)
        return get_best_n_features(avg_ranks, keep_n)

    def _create_outer_loop_results(
        self,
        raw_feature_elim_results: Dict[tuple, InnerLoopResults],
        input_data: InputDataset,
        outer_split: Split,
        data_splitter: DataSplitter,
    ) -> OuterLoopResults:
        feature_elimination_results = self.post_processor.process_feature_elim_results(
            raw_feature_elim_results
        )
        min_eval, mid_eval, max_eval = self._evaluate_min_mid_and_max_features(
            input_data,
            feature_elimination_results.best_features,
            outer_split,
            data_splitter,
        )
        outer_loop_results = OuterLoopResults(
            min_eval=min_eval,
            mid_eval=mid_eval,
            max_eval=max_eval,
            n_features_to_score_map=feature_elimination_results.n_features_to_score_map,
        )
        return outer_loop_results

    def _evaluate_min_mid_and_max_features(
        self,
        input_data: InputDataset,
        best_features: SelectedFeatures,
        split: Split,
        data_splitter: DataSplitter,
    ) -> Tuple[
        FeatureEvaluationResults, FeatureEvaluationResults, FeatureEvaluationResults
    ]:
        min_feats = best_features.min_feats
        mid_feats = best_features.mid_feats
        max_feats = best_features.max_feats

        data_min_feats = data_splitter.split_data(input_data, split, min_feats)
        data_mid_feats = data_splitter.split_data(input_data, split, mid_feats)
        data_max_feats = data_splitter.split_data(input_data, split, max_feats)

        min_eval = self.feature_evaluator.evaluate_features(data_min_feats, min_feats)
        mid_eval = self.feature_evaluator.evaluate_features(data_mid_feats, mid_feats)
        max_eval = self.feature_evaluator.evaluate_features(data_max_feats, max_feats)

        return min_eval, mid_eval, max_eval

    def _select_best_features(
        self, repetition_results: FeatureSelectionResults
    ) -> SelectedFeatures:
        self.results = self._fetch_results(repetition_results)
        selected_features = self.post_processor.select_features(self.results)
        return selected_features

    def _fetch_results(
        self, results: FeatureSelectionResults
    ) -> FeatureSelectionResults:

        log.info("Retrieving results...")
        with progressbar.ProgressBar(max_value=self.n_repetitions * self.n_outer) as b:
            progress = 0
            b.update(progress)

            fetched_results = []
            for repetition in results:
                ol_results = []

                for outer_loop_result in repetition:
                    fetched_outer_loop = outer_loop_result.result()
                    ol_results.append(fetched_outer_loop)

                    progress += 1
                    b.update(progress)

                fetched_results.append(ol_results)
        return fetched_results

    def get_validation_curves(self) -> Dict[str, List]:
        """
        Refer to post_processor.PostProcessor.get_validation_curves for documentation
        """
        return self.post_processor.get_validation_curves(self.results)

    def get_selected_features(
        self, feature_names: List[str] = None
    ) -> SelectedFeatures:
        """Retrieve the selected feature for the three models. Features are normally
        returned as 0-based integer indices representing the columns of the input
        predictor variables (X), however if a list of feature names is provided via
        `feature_names`, the feature names are returned instead.


        Parameters
        ----------
        feature_names : List[str], optional
            the name of every feature, by default None

        Returns
        -------
        SelectedFeatures
            The features selected by the double CV loops

        Raises
        ------
        NotFitException
            if the `fit` method was not called successfully already
        """

        if not self.is_fit:
            raise NotFitException("The feature selector is not fit yet")

        if feature_names is not None:
            if len(feature_names) != self._n_features:
                raise ValueError(
                    f"feature_names provided should contain {self._n_features} elements"
                )
            min_names = [feature_names[f] for f in self._selected_features.min_feats]
            mid_names = [feature_names[f] for f in self._selected_features.mid_feats]
            max_names = [feature_names[f] for f in self._selected_features.max_feats]

            return SelectedFeatures(
                min_feats=min_names,
                max_feats=max_names,
                mid_feats=mid_names,
            )

        return SelectedFeatures(
            min_feats=self._selected_features.min_feats[:],
            max_feats=self._selected_features.max_feats[:],
            mid_feats=self._selected_features.mid_feats[:],
        )

    def make_report(self, feature_names: List[str]) -> SelectedFeatures:
        """
        Prints a small report of the results obtained from the feature selection.

        Parameters
        ----------
        feature_names: List[str]
            List with the name of the features of the original data

        Returns
        -------
        SelectedFeatures:
            The selected features obtained from running the algorithm

        """
        selected_features = self.get_selected_features(feature_names)
        self._print_report(selected_features)

        return selected_features

    @staticmethod
    def _print_report(selected_features: SelectedFeatures):
        print(f"Min features ({len(selected_features.min_feats)}): "
              f"{', '.join(selected_features.min_feats)}\n")
        print(f"Mid features ({len(selected_features.mid_feats)}): "
              f"{', '.join(selected_features.mid_feats)}\n")
        print(f"Max features ({len(selected_features.max_feats)}): "
              f"{', '.join(selected_features.max_feats)}\n")

    def __repr__(self):
        fs = (
            f"FeatureSelector("
            f"repetitions={self.n_repetitions},"
            f" n_outer={self.n_outer},"
            f" n_inner={self.n_inner},"
            f" keep_fraction={self.keep_fraction},"
            f" is_fit={self.is_fit})"
        )

        return fs
