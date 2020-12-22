from typing import List, Dict, Callable, Tuple, TypeVar, Union
from scipy.stats import gmean
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

MIN = "min"
MAX = "max"
MID = "mid"

NumpyArray = np.ndarray
MetricFunction = Callable[[NumpyArray, NumpyArray], float]
Split = Tuple[NumpyArray, NumpyArray]
GenericEstimator = TypeVar("GenericEstimator")
Estimator = Union[BaseEstimator, GenericEstimator]

def compute_number_of_features(
    scores: List[Dict[int, float]], robust_minimum: float
) -> Dict[str, int]:
    """Compute the min, max and mid number of features from a list of fitness
    scores. The scores are averaged, and normalized between 0 (minimum) and 1
    (maximum). Then all the number of features having scores smaller than
    self.robust_minimum are considered. The three values are the minum and the
    maximum of this list and their geometrical mean.
    """
    avg_score = average_scores(scores)
    norm_score = _normalize_score(avg_score)
    n_feats_close_to_minumum = [n for n, s in norm_score.items() if s <= robust_minimum]
    max_feats = max(n_feats_close_to_minumum)
    min_feats = min(n_feats_close_to_minumum)
    mid_feats = int(round(gmean([max_feats, min_feats])))
    return {
        MIN: min_feats,
        MAX: max_feats,
        MID: mid_feats,
    }


def average_scores(scores: List[Dict]) -> Dict[int, float]:
    avg_score = pd.DataFrame(scores).fillna(0).mean().to_dict()
    return avg_score


def _normalize_score(score: Dict[int, float]) -> Dict[int, float]:
    """Normalize input score between min (=0) and max (=1)"""
    max_s = max(score.values())
    min_s = min(score.values())
    delta = max_s - min_s if max_s != min_s else 1
    return {key: (val - min_s) / delta for key, val in score.items()}
