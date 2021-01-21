import pytest
from omigami.plot_utils import plot_validation_curves, plot_feature_rank
from omigami.feature_selector import FeatureSelector


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

