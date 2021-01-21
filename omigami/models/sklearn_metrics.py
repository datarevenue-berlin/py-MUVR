from typing import List
from sklearn.metrics import (
    r2_score,
    median_absolute_error,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    explained_variance_score,
    jaccard_score,
)


class SklearnMetricsWrapper:
    def __init__(self, score_function, greater_is_better=True, **kwargs):
        self.score_func = score_function
        self.kwargs = kwargs
        self.sign = 1 if greater_is_better else -1
        self.greater_is_better = greater_is_better

    def __call__(self, y_true, y_pred):
        return self.sign * self.score_func(y_true, y_pred, **self.kwargs)

    def __repr__(self):
        return (
            f"SklearnMetricsWrapper(score_function={self.score_func}, "
            + f"greater_is_better={self.greater_is_better})"
        )


_SKLEARN_REGRESSION_METRICS = {
    "explained_variance": SklearnMetricsWrapper(explained_variance_score),
    "r2": SklearnMetricsWrapper(r2_score),
    "max_error": SklearnMetricsWrapper(max_error, greater_is_better=False),
    "neg_median_absolute_error": SklearnMetricsWrapper(
        median_absolute_error, greater_is_better=False
    ),
    "neg_mean_absolute_error": SklearnMetricsWrapper(
        mean_absolute_error, greater_is_better=False
    ),
    "neg_mean_squared_error": SklearnMetricsWrapper(
        mean_squared_error, greater_is_better=False
    ),
    "neg_mean_squared_log_error": SklearnMetricsWrapper(
        mean_squared_log_error, greater_is_better=False
    ),
    "neg_root_mean_squared_error": SklearnMetricsWrapper(
        mean_squared_error, greater_is_better=False, squared=False
    ),
    "neg_mean_poisson_deviance": SklearnMetricsWrapper(
        mean_poisson_deviance, greater_is_better=False
    ),
    "neg_mean_gamma_deviance": SklearnMetricsWrapper(
        mean_gamma_deviance, greater_is_better=False
    ),
}

_SKLEARN_CLASSIFICATION_METRICS = {
    "accuracy": SklearnMetricsWrapper(accuracy_score),
    "balanced_accuracy": SklearnMetricsWrapper(balanced_accuracy_score),
    "precision": SklearnMetricsWrapper(precision_score, average="binary"),
    "precision_macro": SklearnMetricsWrapper(
        precision_score, pos_label=None, average="macro"
    ),
    "precision_micro": SklearnMetricsWrapper(
        precision_score, pos_label=None, average="micro"
    ),
    "recall": SklearnMetricsWrapper(recall_score, average="binary"),
    "recall_macro": SklearnMetricsWrapper(
        recall_score, pos_label=None, average="macro"
    ),
    "recall_micro": SklearnMetricsWrapper(
        recall_score, pos_label=None, average="micro"
    ),
    "f1": SklearnMetricsWrapper(f1_score, average="binary"),
    "f1_macro": SklearnMetricsWrapper(f1_score, pos_label=None, average="macro"),
    "f1_micro": SklearnMetricsWrapper(f1_score, pos_label=None, average="micro"),
    "jaccard": SklearnMetricsWrapper(jaccard_score, average="binary"),
    "jaccard_macro": SklearnMetricsWrapper(
        jaccard_score, pos_label=None, average="macro"
    ),
    "jaccard_micro": SklearnMetricsWrapper(
        jaccard_score, pos_label=None, average="micro"
    ),
}


SKLEARN_METRICS = _SKLEARN_CLASSIFICATION_METRICS.copy()
SKLEARN_METRICS.update(_SKLEARN_REGRESSION_METRICS)


def get_supported_classification_metrics() -> List[str]:
    return list(_SKLEARN_CLASSIFICATION_METRICS.keys())


def get_supported_regression_metrics() -> List[str]:
    return list(_SKLEARN_REGRESSION_METRICS.keys())
