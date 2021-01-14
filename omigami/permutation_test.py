from typing import Iterable, Tuple, List
from concurrent.futures import Executor
import numpy as np
import scipy
import tqdm
from omigami.utils import compute_t_student_p_value
from omigami.data_structures.data_types import NumpyArray
from omigami.feature_selector import FeatureSelector


class PermutationTest:
    """"Implements a permutation test of the omigami feature selection.
    The tests fits the input `feature_selector` `n_permutations` times.
    The output vector `y` is scambled at every iteration so that the original
    feature selection results can be contrasted against the results of the
    randomized ones. Statistical significance is addressed using a t-Test.

    The resuts of the randomized fit can be inspected using `compute_permutation_scores`
    If the scores returned by the permutations are very non-normally distributed,
    it's probably better to invoke `compute_p_values` with `ranks=True`, to mitigate
    the violation of the normality assumption needed to perform a meaningful t-Test.

    NOTE: for efficiency, if the input feature selector has been fit already,
    it will not be fit again against the original X, y pair.

    Parameters
    ----------
    feature_selector: FeatureSelector
        the FeatureSelector instance to be used
    n_permutations: int
        number of times the fit on shuffled `y`is repeated

    Examples
    --------
    >>> fs = FeatureSelector(n_outer=6, metric="MISS", estimator="PLSC")
    >>> pt = PermutationTest(fs, 10)
    >>> pt.fit(X, y)
    >>> pt.compute_p_values(model="min")

    """

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

        if not self._fs.is_fit:
            self._fs.fit(X, y, groups=groups)
        self.res = self._get_feats_and_scores()

        fs_perm = FeatureSelector(**self._fs.get_params())
        self.res_perm = []
        for _ in tqdm.tqdm(range(self.n_permutations)):
            np.random.shuffle(y_idx)
            y_perm = y[y_idx]
            fs_perm.fit(X, y_perm, groups=groups, executor=executor)
            self.res_perm.append(self._get_feats_and_scores())

        return self

    def _get_feats_and_scores(self) -> tuple:
        selected_features = self._fs.get_selected_features()
        score = self._fs.get_validation_curves()["total"][0]
        return selected_features, score

    def compute_permutation_scores(self, model: str) -> Tuple[float, List[float]]:
        """Compute the permutation tests scores for input model. `model` must be one of
        `"min"`, `"mid"` or `"max"`. It returns a tuple. The first element is the score
        of the original featur selection, the second element is a list of scores, one
        for each permutation

        Parameters
        ----------
        model : str
            one of `"min"`, `"mid"` or `"max"` models

        Returns
        -------
        tuple
            the original feature selection score and the scores for every permutation

        Raises
        ------
        RuntimeError
            if `fit` is not called first
        ValueError
            if `model` is not one of the three values allowed
        """
        if not self.res:
            raise RuntimeError("Call fit method first")
        if model not in {"min", "mid", "max"}:
            return ValueError("Input model must be one of 'min', 'mid' or 'max'")
        x = self._compute_model_score(model, self.res)
        x_perm = [self._compute_model_score(model, r) for r in self.res_perm]
        return x, x_perm

    def compute_p_values(self, model: str, ranks: bool = False) -> float:
        """Compute the p-value relative to the original feature selection score for
        the input `model` based on the results of the various permutations.
        If the permutation scores are not normal, set `ranks=True`.

        Parameters
        ----------
        model : str
            one of `"min"`, `"mid"` or `"max"` models
        ranks : bool, optional
            whether to perform the significance tests on the ranks rather than on
            the actual scores, by default False

        Returns
        -------
        float
            the permutation test p-value
        """
        if model not in {"min", "mid", "max"}:
            return ValueError("Input model must be one of 'min', 'mid' or 'max'")
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
