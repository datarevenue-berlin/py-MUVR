from omigami.plot_utils import plot_validation_curves
from omigami.omigami import FeatureSelector


def test_plot_validation_curves(results):
    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="RFC",)
    fs.n_features = 10
    fs.is_fit = True
    fs.selected_features = {"min": (0, 1), "max": (0, 1), "mid": (0, 1)}
    fs._results = results
    sel_feats = fs._process_results(results)
    fs._selected_features = sel_feats
    ax = plot_validation_curves(fs)
    assert ax
