from typing import Iterable
from concurrent.futures import Executor
import numpy as np
import scipy
import tqdm
from omigami.utils import compute_t_student_p_value
from omigami.data_structures.data_types import NumpyArray
from omigami.feature_selector import FeatureSelector


class PermutationTest:
    def __init__(self, feature_selector: FeatureSelector, n_permutations: int = 20):
        self.n_permutations = n_permutations
        self._fs = feature_selector
        self.res = None
        self.res_perm = None

    def fit(
        self,
        X: NumpyArray,
        y: NumpyArray,
        groups: NumpyArray = None,
        executor: Executor = None,
    ):

        y_idx = np.arange(y.size)

        self._fs.fit(X, y, groups=groups)
        self.res = self._get_feats_and_scores()

        self.res_perm = []
        for _ in tqdm.tqdm(range(self.n_permutations)):
            np.random.shuffle(y_idx)
            y_perm = y[y_idx]
            self._fs.fit(X, y_perm, groups=groups, executor=executor)
            self.res_perm.append(self._get_feats_and_scores())

        return self

    def _get_feats_and_scores(self) -> tuple:
        selected_features = self._fs.get_selected_features()
        score = self._fs.get_validation_curves()["total"][0]
        return selected_features, score

    def compute_permutation_scores(self, model):
        if not self.res:
            raise RuntimeError("Call fit method first")
        x = self._compute_model_score(model, self.res)
        x_perm = [self._compute_model_score(model, r) for r in self.res_perm]
        return x, x_perm

    def compute_p_values(self, model, ranks: bool = False):
        x, x_perm = self.compute_permutation_scores(model)

        if ranks:
            x, x_perm = self._rank_data(x, x_perm)

        p_values = compute_t_student_p_value(x, x_perm)
        return p_values

    @staticmethod
    def _compute_model_score(model, feats_and_scores):
        selected_features, scores = feats_and_scores
        features = {
            "min": selected_features.min_feats,
            "max": selected_features.max_feats,
            "mid": selected_features.mid_feats,
        }[model]
        n_feats = len(features)
        if model == "mid":
            # mid is the geomteric mean, the score curve might not contain it
            n_feats = min(scores.n_features, key=lambda x: abs(x - n_feats))
        return scores.scores[scores.n_features.index(n_feats)]

    @staticmethod
    def _rank_data(sample: float, population: Iterable):
        ranks = scipy.stats.rankdata([sample] + list(population))
        return ranks[0], ranks[1:]
