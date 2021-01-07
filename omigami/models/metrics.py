from typing import Union
from sklearn.metrics import SCORERS, get_scorer
from data.data_types import MetricFunction, NumpyArray


def make_metric(metric: Union[str, MetricFunction]) -> MetricFunction:
    """Build metric function using the input `metric`. If a metric is a string
    then is interpreted as a scikit-learn metric score, such as "accuracy".
    Else, if should be a callable accepting two input arrays."""
    if isinstance(metric, str):
        return _make_metric_from_string(metric)
    if hasattr(metric, "__call__"):
        return metric
    raise ValueError("Input metric is not valid")


def _make_metric_from_string(metric_string: str) -> MetricFunction:
    if metric_string == "MISS":
        return miss_score
    if metric_string in SCORERS:
        # pylint: disable=protected-access
        return get_scorer(metric_string)._score_func
    raise ValueError("Input metric is not a valid string")


def miss_score(y_true: NumpyArray, y_pred: NumpyArray):
    """MISS score: number of wrong classifications preceded by - so that the higher
    this score the better the model"""
    return -(y_true != y_pred).sum()
