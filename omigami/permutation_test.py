import logging
from typing import Iterable
import numpy as np
import dask
import scipy
import tqdm
from omigami.utils import compute_t_student_p_value, MIN, MAX
from omigami.omigami import FeatureSelector


class PermutationTest:
    def __init__(self, feature_selector: FeatureSelector, n_permutations: int = 100):
        self.n_permutations = n_permutations
        self._fs = feature_selector
        self.res = None
        self.res_perm = None

    def _get_avg_score(self, model: str) -> float:
        n_feats = len(self._fs.selected_features[model])
        return np.average(
            [
                score[n_feats]
                for ol in self._fs.outer_loop_aggregation
                for score in ol["scores"]
            ]
        )

    def _get_avg_scores(self) -> dict:
        return {model: self._get_avg_score(model) for model in {MIN, MAX}}

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):

        dask.delayed(X)
        y_idx = np.arange(len(y))

        self._fs.fit(X, y, groups=groups)

        self.res = self._get_avg_scores()
        logging.debug("Run permutations")
        self.res_perm = []
        for _ in tqdm.tqdm(range(self.n_permutations)):
            np.random.shuffle(y_idx)
            y_perm = y[y_idx]

            self._fs.fit(X, y_perm, groups=groups)

            self.res_perm.append(self._get_avg_scores())

        return self

    def compute_p_values(self, ranks: bool = False):
        if not self.res:
            raise RuntimeError("Call fit method first")
        p_values = {}
        for model in {MIN, MAX}:
            x = self.res[model]
            x_perm = [r[model] for r in self.res_perm]
            if ranks:
                x, x_perm = self._rank_data(x, x_perm)
            p_values[model] = compute_t_student_p_value(x, x_perm)
        return p_values

    @staticmethod
    def _rank_data(sample: float, population: Iterable):
        ranks = scipy.stats.rankdata([sample] + list(population))
        return ranks[0], ranks[1:]
