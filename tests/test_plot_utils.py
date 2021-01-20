from unittest.mock import Mock
import pytest
from omigami.plot_utils import (
    plot_validation_curves,
    plot_feature_rank,
    plot_permutation_scores,
)
from omigami.feature_selector import FeatureSelector
from omigami.permutation_test import PermutationTest


@pytest.fixture
def fit_feature_selector(results):
    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="RFC")
    fs.results = results
    fs.is_fit = True
    fs._selected_features = fs.post_processor.select_features(results)
    return fs


@pytest.fixture
def permutation_test():
    pt = Mock(PermutationTest)
    pt.compute_permutation_scores = Mock(
        spec=pt.compute_permutation_scores, return_value=(1, list(range(2, 1000)))
    )
    pt.compute_p_values = Mock(spec=pt.compute_p_values, return_value=0.01)
    return pt


def test_plot_validation_curves(fit_feature_selector):
    ax = plot_validation_curves(fit_feature_selector)
    assert ax


@pytest.mark.parametrize("model", ["min", "max", "mid"])
def test_plot_feature_rank(fit_feature_selector, model):
    fig = plot_feature_rank(fit_feature_selector, model)
    assert fig


def test_plot_feature_rank_error(fit_feature_selector):
    with pytest.raises(ValueError):
        fig = plot_feature_rank(fit_feature_selector, "yo")


def test_plot_permutiation_scores(permutation_test):
    fig = plot_permutation_scores(permutation_test, "min")
    assert fig
