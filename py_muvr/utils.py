from typing import List, Dict, Iterable, Callable
import logging
import pandas as pd
import numpy as np
import scipy.stats
from py_muvr.data_structures import FeatureRanks


def average_scores(scores: List[Dict]) -> Dict[int, float]:
    avg_score = pd.DataFrame(scores).fillna(0).mean().to_dict()
    return avg_score


def normalize_score(score: Dict[int, float]) -> Dict[int, float]:
    """Normalize input score between min (=0) and max (=1)"""
    max_s = max(score.values())
    min_s = min(score.values())
    delta = max_s - min_s if max_s != min_s else 1
    return {key: (val - min_s) / delta for key, val in score.items()}


def average_ranks(ranks: Iterable[FeatureRanks]) -> FeatureRanks:
    n_feats = set(r.n_feats for r in ranks)
    if len(n_feats) > 1:
        raise ValueError("Input ranks refer to different features")
    n_feats = n_feats.pop()
    features = np.arange(n_feats)
    avg_ranks = []
    for f in features:
        avg_rank = np.average([rank[f] for rank in ranks])
        avg_ranks.append(avg_rank)
    return FeatureRanks(features=features, ranks=avg_ranks)


def get_best_n_features(ranks: FeatureRanks, n_to_keep: int) -> List[int]:
    ranks_data = ranks.get_data()
    sorted_data = sorted(ranks_data.items(), key=lambda x: x[1])
    feats = [feat for feat, _ in sorted_data[0:n_to_keep]]

    if len(feats) == n_to_keep:
        return feats

    # pad with non-present features, scramble to not introduce a bias
    all_feats = np.arange(ranks.n_feats)
    np.random.shuffle(all_feats)
    for f in all_feats:
        if f not in ranks_data:
            feats.append(f)
            if len(feats) == n_to_keep:
                return feats

    raise ValueError("Impossible to return so many best features")


def compute_t_student_p_value(sample: float, population: Iterable) -> float:
    """From Wikipeida:
    https://en.wikipedia.org/wiki/Prediction_interval#Unknown_mean,_unknown_variance
    Compute the p_value of the t-Student variable that represent the distribution
    of the n+1 sample of population, where `n = len(population)`.
    In few words, it gives the probability that `sample` belongs to `population`.
    The underlying assumption is that population is normally distributed
    with unknown mean and unknown variance

    Parameters
    ----------
        sample: float
            value for which we want the p-value
        population: Iterable
            values that represent the null hypothesis for `sample`

    Returns
    -------
        float
            the t-Student test p-value
    """
    m = np.mean(population)
    s = np.std(population)
    n = len(population)
    t = (sample - m) / (s * (1 + 1 / n) ** 0.5)
    return scipy.stats.t(df=n - 1).cdf(t)


def mute_loggers(loggers: List[str] = None):
    """Decorator to mute specific loggers within a function call. This is particularly
    useful when nested for loops might add excessive verbosity to a specific method.
    The input `loggers` is a list of loggers by name that should be muted. The logging
    level of the selected loggers is set to WARNING and restored to the original value
    at the end of the decorated method execution.

    Parameters
    ----------
    loggers : List[str], optional
        List of loggers by name that have to be muted, by default None
    """
    loggers = loggers or []

    def decorator(function: Callable):
        def decorated(*args, **kwargs):
            logs = {log_name: logging.getLogger(log_name) for log_name in loggers}
            old_levels = {
                log_name: log.getEffectiveLevel() for log_name, log in logs.items()
            }
            for log in logs.values():
                log.setLevel(logging.WARN)
            try:
                res = function(*args, **kwargs)
            finally:
                for log_name, log in logs.items():
                    log.setLevel(old_levels[log_name])
            return res

        return decorated

    return decorator
