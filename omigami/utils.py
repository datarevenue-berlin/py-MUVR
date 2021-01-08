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
    ranks_data = ranks.get_data()
    sorted_data = sorted(ranks_data.items(), key=lambda x: x[1])
    feats = [feat for feat, _ in sorted_data[0:n]]

    if len(feats) == n:
        return feats

    # pad with non-present features, scrumble to not introduce a bias
    all_feats = np.arange(ranks.n_feats)
    np.random.shuffle(all_feats)
    for f in all_feats:
        if f not in ranks_data:
            feats.append(f)
            if len(feats) == n:
                return feats

    raise ValueError("Impossible to return so many best features")
