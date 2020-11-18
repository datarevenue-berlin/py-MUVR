import pytest
import matplotlib.pyplot as plt
from omigami import plot_utils
from omigami.omigami import FeatureSelector


@pytest.fixture
def fitted_selector(results):
    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="RFC",)
    fs.n_features = 10
    fs.is_fit = True
    fs.selected_features = {"min": {0, 1}, "max": {0, 1}, "mid": {0, 1}}
    fs.results = results
    sel_feats = fs._process_results(results)
    fs._selected_features = sel_feats
    return fs


def test_plot_validation_curves(fitted_selector):
    ax = plot_utils.plot_validation_curves(fitted_selector)
    assert ax


def test_plot_feature_rank(fitted_selector):
    fig = plot_utils.plot_feature_rank(fitted_selector, "min")
    assert isinstance(fig, plt.Figure)
