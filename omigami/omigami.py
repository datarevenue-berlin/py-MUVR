import logging
from typing import Union, Callable, List, Dict, Tuple, TypeVar
import numpy as np
import pandas as pd
from scipy.stats import gmean, rankdata
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import sklearn.metrics
import dask
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

NumpyArray = np.ndarray
MetricFunction = Callable[[NumpyArray, NumpyArray], float]
Split = Tuple[NumpyArray, NumpyArray]
GenericEstimator = TypeVar("GenericEstimator")
Estimator = Union[BaseEstimator, GenericEstimator]


class FeatureSelector:
    RFC = "RFC"
    MIN = "min"
    MAX = "max"
    MID = "mid"

    def __init__(
        self,
        X: NumpyArray,
        y: NumpyArray,
        n_outer: int,
        metric: Union[str, MetricFunction],
        estimator: Union[str, Estimator],
        features_dropout_rate: float = 0.05,
        robust_minimum: float = 0.05,
        n_inner: int = None,
        groups: NumpyArray = None,
        repetitions: int = 8,
    ):
        self.X = X
        self.y = y
        self.n_outer = n_outer
        self.metric = self._make_metric(metric)
        self.estimator = self._make_estimator(estimator)
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
        results_futures = []
        for j in range(self.repetitions):
            splits = self._make_splits()
            results_futures.append(
                [self._perform_outer_loop_cv(i, splits) for i in range(self.n_outer)]
            )
        results = dask.compute(
            results_futures,
            # scheduler="single-threaded"
        )[0]
        self._results = results
        self._selected_features = self._process_results(results)
        return self._selected_features

    def _process_results(self, results: List) -> Dict[str, set]:
        outer_loop_aggregation = [self._process_outer_loop(ol) for ol in results]
        return self._select_best_features(outer_loop_aggregation)

    def _process_outer_loop(self, outer_loop_results: List) -> Dict:
        avg_feature_rank = self._compute_avg_feature_rank(outer_loop_results)
        scores = [r["scores"] for r in outer_loop_results]
        n_feats = self._compute_number_of_features(scores)
        return {
            "avg_feature_ranks": avg_feature_rank,
            "scores": scores,
            "n_feats": n_feats,
        }

    @dask.delayed
    def _perform_outer_loop_cv(self, i: int, splits: Dict[tuple, Split]) -> Dict:
        outer_train_results = self._perform_inner_loop_cv(i, splits)
        res = self._select_best_features_and_score(outer_train_results)
        scores = res.pop("score")
        outer_test_results = {
            key: self._train_and_evaluate_on_segments(splits[(i,)], features)
            for key, features in res.items()
        }
        return {
            "test_results": outer_test_results,
            "scores": scores,
        }

    def _compute_avg_feature_rank(
        self, outer_loop_results: List
    ) -> Dict[str, pd.DataFrame]:
        outer_test_results = [r["test_results"] for r in outer_loop_results]
        avg_feature_rank = {}
        for key in {self.MIN, self.MAX, self.MID}:
            feature_ranks = [res[key]["feature_ranks"] for res in outer_test_results]
            avg_feature_rank[key] = (
                pd.DataFrame(feature_ranks).fillna(0).mean().to_dict()
            )
        return avg_feature_rank

    def _compute_number_of_features(
        self, scores: List[Dict[int, float]]
    ) -> Dict[str, int]:
        avg_score = self._average_scores(scores)
        norm_score = self._normalize_score(avg_score)
        max_feats = max(n for n, s in norm_score.items() if s <= self.robust_minimum)
        min_feats = min(n for n, s in norm_score.items() if s <= self.robust_minimum)
        mid_feats = int(round(gmean([max_feats, min_feats])))
        return {
            self.MIN: min_feats,
            self.MAX: max_feats,
            self.MID: mid_feats,
        }

    @staticmethod
    def _average_scores(scores: List[Dict]) -> Dict[int, float]:
        avg_score = pd.DataFrame(scores).fillna(0).mean().to_dict()
        return avg_score

    @staticmethod
    def _normalize_score(score: Dict[int, float]) -> Dict[int, float]:
        max_s = max(score.values())
        min_s = min(score.values())
        delta = max_s - min_s if max_s != min_s else 1
        return {key: (val - min_s) / delta for key, val in score.items()}

    def _perform_inner_loop_cv(
        self, i: int, splits: Dict[tuple, Split]
    ) -> Dict[Tuple[int], List]:
        final_results = {}
        features = list(range(self.n_features))
        while len(features) > 1:
            inner_results = []
            for j in range(self.n_inner):
                split_id = (i, j)
                split = splits[split_id]
                results = self._train_and_evaluate_on_segments(split, features)
                inner_results.append(results)
            final_results[tuple(features)] = inner_results
            features = self._keep_best_features(inner_results, features)
        return final_results

    def _train_and_evaluate_on_segments(
        self, split: Split, features: List[int]
    ) -> Dict:
        inner_train_idx, inner_test_idx = split
        X_train = self.X[inner_train_idx, :][:, features]
        X_test = self.X[inner_test_idx, :][:, features]
        y_train = self.y[inner_train_idx]
        y_test = self.y[inner_test_idx]
        model = clone(self.estimator)
        y_pred = model.fit(X_train, y_train).predict(X_test)
        feature_ranks = self._extract_feature_rank(model, features)
        return {
            "score": -self.metric(y_pred, y_test),
            "feature_ranks": feature_ranks,
        }

    def _keep_best_features(
        self, inner_results: List, features: List[int]
    ) -> List[int]:
        feature_ranks = [r["feature_ranks"] for r in inner_results]
        avg_ranks = pd.DataFrame(feature_ranks).fillna(self.n_features).mean().to_dict()
        for f in features:
            if f not in avg_ranks:
                avg_ranks[f] = self.n_features
        sorted_averages = sorted(avg_ranks.items(), key=lambda x: x[1])
        n_features_to_drop = round(self.features_dropout_rate * len(features))
        if not n_features_to_drop:
            n_features_to_drop = 1
        sorted_averages = sorted_averages[:-n_features_to_drop]
        return [feature for feature, _ in sorted_averages]

    def _extract_feature_rank(
        self, estimator: Estimator, features: List[int]
    ) -> Dict[int, float]:
        if hasattr(estimator, "feature_importances_"):
            ranks = rankdata(-estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            ranks = rankdata(-np.abs(estimator.coef_[0]))
        else:
            raise ValueError("The estimator provided has no feature importances")
        return dict(zip(features, ranks))

    def _select_best_features_and_score(
        self, outer_train_results: Dict[Tuple[int], Dict]
    ) -> Dict:
        features_kept = {}
        score = {}
        for features, res in outer_train_results.items():
            n_feats = len(features)
            features_kept[n_feats] = features
            score[n_feats] = np.sum([r["score"] for r in res])

        n_feats = self._compute_number_of_features([score])
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

    def _select_best_features(self, results: List) -> Dict[str, set]:
        final_feature_ranks = self._compute_final_ranks(results)
        avg_scores = [self._average_scores(r["scores"]) for r in results]
        n_feats = self._compute_number_of_features(avg_scores)
        feature_sets = {}
        for key in (self.MIN, self.MAX, self.MID):
            feats = final_feature_ranks.sort_values(by=key).head(n_feats[key])
            feature_sets[key] = set(feats[key].index)
        return feature_sets

    def _compute_final_ranks(self, results: List) -> pd.DataFrame:
        final_ranks = {}
        for key in (self.MIN, self.MAX, self.MID):
            avg_ranks = [r["avg_feature_ranks"][key] for r in results]
            final_ranks[key] = (
                pd.DataFrame(avg_ranks).fillna(self.n_features).mean().to_dict()
            )
        return pd.DataFrame.from_dict(final_ranks).fillna(self.n_features)

    def _make_estimator(self, estimator: Union[str, Estimator]) -> Estimator:
        if estimator == self.RFC:
            return RandomForestClassifier(n_estimators=150)
        elif isinstance(estimator, BaseEstimator):
            return estimator
        else:
            raise ValueError("Unknown type of estimator")

    def _make_splits(self) -> Dict[tuple, Split]:
        outer_splitter = GroupKFold(self.n_outer)
        inner_splitter = GroupKFold(self.n_inner)
        outer_splits = outer_splitter.split(self.X, self.y, self.groups)
        splits = {}
        for i, (out_train, out_test) in enumerate(outer_splits):
            splits[(i,)] = out_train, out_test
            inner_splits = inner_splitter.split(
                self.X[out_train, :], self.y[out_train], self.groups[out_train]
            )
            for j, (inner_train, inner_valid) in enumerate(inner_splits):
                splits[(i, j)] = out_train[inner_train], out_train[inner_valid]
        return splits

    def _make_metric(self, metric: Union[str, MetricFunction]):
        if isinstance(metric, str):
            return self._make_metric_from_string(metric)
        elif hasattr(metric, "__call__"):
            return metric
        else:
            raise ValueError("Input metric is not valid")

    @staticmethod
    def _make_metric_from_string(metric_string: str) -> MetricFunction:
        if metric_string == "MISS":
            return miss_score
        elif metric_string in sklearn.metrics.SCORERS:
            return sklearn.metrics.get_scorer(metric_string)._score_func
        else:
            raise ValueError("Input metric is not a valid string")

    def plot_validation_curves(self) -> Axes:
        if self._results is None or self._selected_features is None:
            logging.warning(
                "Validation curves have not been generated. To be able to plot call `select_features` method first"
            )
        outer_loop_aggregation = [self._process_outer_loop(ol) for ol in self._results]
        for res in outer_loop_aggregation:
            for i, score in enumerate(res["scores"]):
                label = "Outer loop average" if i == 0 else None
                sorted_score_items = sorted(score.items())
                n_feats, score_values = zip(*sorted_score_items)
                plt.semilogx(n_feats, score_values, c="#deebf7", label=label)
        repetition_averages = []
        for i, r in enumerate(outer_loop_aggregation):
            label = "Repetition average" if i == 0 else None
            avg_scores = self._average_scores(r["scores"])
            sorted_score_items = sorted(avg_scores.items())
            n_feats, score_values = zip(*sorted_score_items)
            plt.semilogx(n_feats, score_values, c="#3182bd", label=label)
            repetition_averages.append(avg_scores)
        final_avg = self._average_scores(repetition_averages)
        sorted_score_items = sorted(avg_scores.items())
        n_feats, score_values = zip(*sorted_score_items)
        plt.semilogx(n_feats, score_values, c="k", lw=3, label="Final average")
        plt.xlabel("# features")
        plt.ylabel("Fitness score")
        plt.grid(ls=":")
        return plt.gca()


def miss_score(y_true, y_pred):
    return -(y_true != y_pred).sum()
