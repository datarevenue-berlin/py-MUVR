from __future__ import annotations

import logging
from concurrent.futures import Executor, Future
from copy import deepcopy
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import progressbar
from numpy.random import RandomState

from py_muvr.data_structures import (
    InputDataset,
    SelectedFeatures,
    OuterLoopResults,
    InnerLoopResults,
    Split,
    FeatureEvaluationResults,
    MetricFunction,
    InputEstimator,
    NumpyArray,
    FeatureSelectionRawResults,
    FeatureSelectionResults,
)
from py_muvr.feature_evaluator import FeatureEvaluator
from py_muvr.data_splitter import DataSplitter
from py_muvr.models import Estimator
from py_muvr.post_processor import PostProcessor
from py_muvr.utils import get_best_n_features, average_ranks
from py_muvr.exceptions import NotFitException
from py_muvr.sync_executor import SyncExecutor

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
        self.n_outer = n_outer
        self.metric = metric
        self.estimator = estimator
        self.features_dropout_rate = features_dropout_rate
        self.robust_minimum = robust_minimum
        self.n_inner = self._set_n_inner(n_inner)
        self.n_repetitions = n_repetitions
        self.random_state = None if random_state is None else RandomState(random_state)

        self.is_fit = False
        self._keep_fraction = 1 - features_dropout_rate
        self._n_features = None
        self._selected_features = None
        self._raw_results = None
        self._minimum_features = 1

        self._feature_evaluator = FeatureEvaluator(estimator, metric, random_state)
        self._post_processor = PostProcessor(robust_minimum)

    @property
    def raw_results(self):
        if self.is_fit:
            return self._raw_results
        else:
            raise NotFitException("The feature selector is not fit yet")

    def _set_n_inner(self, n_inner: Union[int, None]) -> int:
        if not n_inner:
            log.info("Parameter n_inner is not specified, setting it to n_outer - 1")
            n_inner = self.n_outer - 1
        return n_inner

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
        self._feature_evaluator.set_n_initial_features(n_features)

        log.info(
            f"Running {self.n_repetitions} repetitions and"
            f" {self.n_outer} outer loops using "
            f"executor {executor.__class__.__name__}."
        )

        repetition_results = []

        log.info("Scheduling tasks...")
        Progressbar = self._make_progress_bar()
        with Progressbar(max_value=self.n_repetitions * self.n_outer) as b:
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

                feature_evaluation_results = self._feature_evaluator.evaluate_features(
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
        features_to_keep = int(np.floor(len(features) * self._keep_fraction))
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
        feature_elimination_results = self._post_processor.process_feature_elim_results(
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
        min_feats = best_features["min"]
        mid_feats = best_features["mid"]
        max_feats = best_features["max"]

        data_min_feats = data_splitter.split_data(input_data, split, min_feats)
        data_mid_feats = data_splitter.split_data(input_data, split, mid_feats)
        data_max_feats = data_splitter.split_data(input_data, split, max_feats)

        min_eval = self._feature_evaluator.evaluate_features(data_min_feats, min_feats)
        mid_eval = self._feature_evaluator.evaluate_features(data_mid_feats, mid_feats)
        max_eval = self._feature_evaluator.evaluate_features(data_max_feats, max_feats)

        return min_eval, mid_eval, max_eval

    def _select_best_features(
        self, repetition_results: FeatureSelectionRawResults
    ) -> SelectedFeatures:
        self._raw_results = self._fetch_results(repetition_results)
        selected_features = self._post_processor.select_features(self._raw_results)
        return selected_features

    def _fetch_results(
        self, results: FeatureSelectionRawResults
    ) -> FeatureSelectionRawResults:

        log.info("Retrieving results...")
        Progressbar = self._make_progress_bar()
        with Progressbar(max_value=self.n_repetitions * self.n_outer) as b:
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

    def get_feature_selection_results(
        self, feature_names: List[str] = None
    ) -> FeatureSelectionResults:
        """
        Retrieve the feature selection results in a single data structure. This object
        contains the attributes:
        - raw_results: selection results before processing
        - selected_features: 0-based integer indices for min, mid and max feature sets
        - score_curves: validation curves of n_feats vs score
        - selected_feature_names: list of feature names if the parameter is used.

        Parameters
        ----------
        feature_names : List[str], optional
            the name of every feature, by default None

        Returns
        -------
        FeatureSelectionResults
            The results obtained from running the algorithm

        Raises
        ------
        NotFitException
            if the `fit` method was not called successfully already
        """
        if not self.is_fit:
            raise NotFitException("The feature selector is not fit yet")

        return FeatureSelectionResults(
            raw_results=deepcopy(self._raw_results),
            selected_features=self.get_selected_features(),
            score_curves=self._get_validation_curves(),
            selected_feature_names=self.get_selected_features(feature_names),
        )

    def _get_selected_feature_names(
        self, feature_names: Union[None, List[str]]
    ) -> Union[None, SelectedFeatures]:
        if feature_names is None:
            return feature_names

        if len(feature_names) != self._n_features:
            raise ValueError(
                f"feature_names provided should contain {self._n_features} elements"
            )
        min_names = [feature_names[f] for f in self._selected_features["min"]]
        mid_names = [feature_names[f] for f in self._selected_features["mid"]]
        max_names = [feature_names[f] for f in self._selected_features["max"]]

        selected_feature_names = SelectedFeatures(
            min=min_names,
            max=max_names,
            mid=mid_names,
        )
        return selected_feature_names

    def get_selected_features(self, feature_names: List[str] = None):
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
        if feature_names is None:
            return deepcopy(self._selected_features)
        else:
            return self._get_selected_feature_names(feature_names)

    def _get_validation_curves(self) -> Dict[str, List]:
        return self._post_processor.get_validation_curves(self._raw_results)

    def __repr__(self):
        fs = (
            f"FeatureSelector("
            f"repetitions={self.n_repetitions},"
            f" n_outer={self.n_outer},"
            f" n_inner={self.n_inner},"
            f" feature_dropout_rate={self.features_dropout_rate},"
            f" is_fit={self.is_fit})"
        )

        return fs

    @staticmethod
    def _make_progress_bar():
        if logging.getLogger(__name__).getEffectiveLevel() > logging.INFO:
            return progressbar.NullBar
        return progressbar.ProgressBar

    def export_average_feature_ranks(
        self,
        output_path: str,
        feature_names: List[str] = None,
        exclude_unused_features: bool = True,
    ) -> pd.DataFrame:
        """
        Creates and saves dataframe from the feature selection results. This dataframe contains
        the columns 'mid', 'mid', and 'max', the indices are the features and the values
        are the average rank across repetitions and outer loops.

        Parameters
        ----------
        output_path: str
            Path where to save the csv dataframe
        feature_names: List[str]
            The name of every feature, by default None
        exclude_unused_features: bool
            Whether to remove the features that werent selected or not.

        Returns
        -------
        pd.DataFrame:
            Pandas dataframe containing the average feature ranks

        """
        ranks_df = self.get_average_ranks_df(feature_names, exclude_unused_features)

        ranks_df.to_csv(output_path)
        return ranks_df

    def get_average_ranks_df(
        self,
        feature_names: List[str] = None,
        exclude_unused_features: bool = True,
    ):
        """
        Creates a dataframe from the feature selection results. This dataframe contains
        the columns 'mid', 'mid', and 'max', the indices are the features and the values
        are the average rank across repetitions and outer loops.

        Parameters
        ----------
        feature_names: List[str]
            The name of every feature, by default None
        exclude_unused_features: bool
            Whether to remove the features that werent selected or not.

        Returns
        -------
        pd.DataFrame:
            Pandas dataframe containing the average feature ranks

        """
        results = self.get_feature_selection_results()
        return self._post_processor.make_average_ranks_df(
            results, self._n_features, feature_names, exclude_unused_features
        )
