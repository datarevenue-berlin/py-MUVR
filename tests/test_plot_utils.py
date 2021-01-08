from omigami.plot_utils import plot_validation_curves
from omigami.feature_selector import FeatureSelector


def test_plot_validation_curves(results):
    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="RFC")
    fs._results = results
    fs.is_fit = True
    fs.selected_features = fs.post_processor.select_features(results)
    ax = plot_validation_curves(fs)
    assert ax
