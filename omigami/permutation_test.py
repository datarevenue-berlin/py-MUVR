import logging
from typing import Iterable
import numpy as np
import dask
import scipy
import tqdm
from omigami.utils import compute_t_student_p_value
from omigami.omigami import FeatureSelectorBase


class PermutationTest:
    def __init__(self, feature_selector: FeatureSelectorBase, n_permutations: int):
        self.n_permutations = n_permutations
        self._fs = feature_selector
        self.res = None
        self.res_perm = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):

        dask.delayed(X)
        y_idx = np.arange(len(y))

        self._fs.fit(X, y, groups=groups)

        self.res = self._fs.selection_score
        logging.debug("Run permutations")
        self.res_perm = []
        for _ in tqdm.tqdm(range(self.n_permutations)):
            np.random.shuffle(y_idx)
            y_perm = y[y_idx]

            self._fs.fit(X, y_perm, groups=groups)

            self.res_perm.append(self._fs.selection_score)

        return self

    def compute_p_values(self, ranks: bool = False):
        if not self.res:
            raise RuntimeError("Call fit method first")
        p_values = {}
        for model in self.res.keys():
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
