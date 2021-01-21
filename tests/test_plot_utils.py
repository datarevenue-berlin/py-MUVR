import pytest
from omigami.plot_utils import plot_validation_curves, plot_feature_rank
from omigami.feature_selector import FeatureSelector


@pytest.fixture
def fit_feature_selector(results):
    fs = FeatureSelector(n_outer=3, metric="MISS", estimator="RFC")
    fs.results = results
    fs.is_fit = True
    fs._selected_features = fs._post_processor.select_features(results)
    return fs


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

