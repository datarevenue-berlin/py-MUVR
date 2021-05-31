import logging
from typing import Iterable, Tuple, List
from concurrent.futures import Executor
import numpy as np
from scipy.stats import rankdata, normaltest
import progressbar
from py_muvr.utils import compute_t_student_p_value, mute_loggers
from py_muvr.data_structures.data_types import NumpyArray
from py_muvr.data_structures.data_models import FeatureSelectionResults
from py_muvr.feature_selector import FeatureSelector

logger = logging.getLogger(__name__)


class PermutationTest:
    """ "Implements a permutation test of the py_muvr feature selection.
    The tests fits the input `feature_selector` `n_permutations` times.
    The target `y` is scrambled at every iteration so that the original
    feature selection results can be contrasted against the results of the
    randomized ones. Statistical significance is addressed using a t-Test.

    The results of the randomized fit can be inspected using
    `compute_permutation_scores`.
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

    @mute_loggers(loggers=["py_muvr.feature_selector", "py_muvr.models.pls"])
    def fit(
        self,
        X: NumpyArray,
        y: NumpyArray,
        groups: NumpyArray = None,
        executor: Executor = None,
    ):
        logger.info("Running permutation test for %s permutations", self.n_permutations)
        y_idx = np.arange(y.size)

        if not self._fs.is_fit:
            self._fs.fit(X, y, groups=groups)
        self.res = self._fs.get_feature_selection_results()

        fs_perm = self._copy_feature_selector(self._fs)
        self.res_perm = []
        for _ in progressbar.progressbar(range(self.n_permutations)):
            np.random.shuffle(y_idx)
            y_perm = y[y_idx]
            fs_perm.fit(X, y_perm, groups=groups, executor=executor)
            self.res_perm.append(fs_perm.get_feature_selection_results())
        logger.info("Finished permutation test. Storing results in self.res_perm")
        return self

    def compute_permutation_scores(self, model: str) -> Tuple[float, List[float]]:
        """Compute the permutation tests scores for input model. `model` must be one of
        `"min"`, `"mid"` or `"max"`. It returns a tuple. The first element is the score
        of the original feature selection, the second element is a list of scores, one
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
            raise ValueError("Input model must be one of 'min', 'mid' or 'max'")
        score = self._compute_model_score(model, self.res)
        score_perm = [self._compute_model_score(model, r) for r in self.res_perm]
        return score, score_perm

    def compute_p_values(self, model: str, ranks: bool = False) -> float:
        """Compute the p-value relative to the original feature selection score for
        the input `model` based on the results of the various permutations.
        If the permutation scores are not normal, `ranks` is automatically set to True.

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
            raise ValueError("Input model must be one of 'min', 'mid' or 'max'")
        score, score_perm = self.compute_permutation_scores(model)

        if not ranks and len(score_perm) > 7:  # or `normaltest` would raise an error
            test = normaltest(score_perm)
            if test.pvalue < 0.05:
                logger.warning(
                    "%s %s",
                    "The permutation scores don't seem to be normally distributed.",
                    "Setting ranks to True (non-parametric test)",
                )
                ranks = True

        if ranks:
            score, score_perm = self._rank_data(score, score_perm)

        p_values = compute_t_student_p_value(score, score_perm)
        return p_values

    @staticmethod
    def _compute_model_score(
        model: str, feats_and_scores: FeatureSelectionResults
    ) -> float:
        selected_features = feats_and_scores.selected_features
        total_score_curve = feats_and_scores.score_curves["total"][0]
        features = getattr(selected_features, model)
        n_feats = len(features)
        if model == "mid":
            # mid is the geometric mean, the score curve might not contain it
            n_feats = min(total_score_curve.n_features, key=lambda x: abs(x - n_feats))
        idx = total_score_curve.n_features.index(n_feats)
        return total_score_curve.scores[idx]

    @staticmethod
    def _rank_data(sample: float, population: Iterable):
        ranks = rankdata([sample] + list(population))
        return ranks[0], ranks[1:]

    @staticmethod
    def _copy_feature_selector(fs: FeatureSelector) -> FeatureSelector:
        return FeatureSelector(
            n_outer=fs.n_outer,
            metric=fs.metric,
            estimator=fs.estimator,
            features_dropout_rate=fs.features_dropout_rate,
            robust_minimum=fs.robust_minimum,
            n_inner=fs.n_inner,
            n_repetitions=fs.n_repetitions,
            random_state=(
                None if fs.random_state is None else fs.random_state.get_state()[1][0]
            ),
        )

    def __repr__(self):
        return f"PermutationTest(n_permutations={self.n_permutations})"
