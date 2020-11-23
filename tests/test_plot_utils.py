from omigami.plot_utils import plot_validation_curves
from omigami.omigami import FeatureSelector


def test_plot_validation_curves(fitted_feature_selector):
    ax = plot_validation_curves(fitted_feature_selector)
    assert ax
