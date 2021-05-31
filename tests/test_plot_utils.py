from unittest.mock import Mock
import pytest
from distributed import Client

from py_muvr import FeatureSelector
from py_muvr.plot_utils import (
    plot_validation_curves,
    plot_feature_rank,
    plot_permutation_scores,
)
from py_muvr.permutation_test import PermutationTest


@pytest.fixture
def permutation_test():
    pt = Mock(PermutationTest)
    pt.compute_permutation_scores = Mock(
        spec=pt.compute_permutation_scores, return_value=(1, list(range(2, 1000)))
    )
    pt.compute_p_values = Mock(spec=pt.compute_p_values, return_value=0.01)
    return pt


def test_plot_validation_curves(fs_results):
    ax = plot_validation_curves(fs_results)
    assert ax


@pytest.mark.parametrize("model", ["min", "max", "mid"])
def test_plot_feature_rank(fs_results, model):
    fig = plot_feature_rank(fs_results, model)
    assert fig


def test_plot_feature_rank_error(fs_results):
    with pytest.raises(ValueError):
        fig = plot_feature_rank(fs_results, "yo")


def test_plot_permutiation_scores(permutation_test):
    fig = plot_permutation_scores(permutation_test, "min")
    assert fig
