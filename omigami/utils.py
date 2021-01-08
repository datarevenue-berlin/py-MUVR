from typing import List, Dict, Iterable
import pandas as pd
import numpy as np
from omigami.data import FeatureRanks


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


def get_best_n_features(ranks: FeatureRanks, n: int) -> List[int]:
    # TODO: improve efficiency of range(avg_ranks.n_feats including an items
    # method to FeatureRanks so that we don't have to go through all n_feats
    # (those that are not there are useless, since they are the highest ranks)
    best_ranks = sorted([(ranks[f], f) for f in range(ranks.n_feats)])
    best_ranks = best_ranks[: int(n)]
    return [f for _, f in best_ranks]
