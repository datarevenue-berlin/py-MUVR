import logging
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from omigami.feature_selector import FeatureSelector
from omigami.utils import average_scores


def plot_validation_curves(feature_selector: FeatureSelector) -> Axes:
    if not feature_selector.is_fit:
        logging.warning(  # pylint: disable=logging-not-lazy
            "Validation curves have not been generated. To be able to plot"
            + " call `select_features` method first"
        )
        return None
    curves = feature_selector.get_validation_curves()
    for i, curve in enumerate(curves["outer_loops"]):
        label = "Outer loop average" if i == 0 else None
        plt.semilogx(curve.n_features, curve.scores, c="#deebf7", label=label)
    for i, curve in enumerate(curves["repetitions"]):
        label = "Repetition average" if i == 0 else None
        plt.semilogx(curve.n_features, curve.scores, c="#3182bd", label=label)
    for i, curve in enumerate(curves["total"]):
        label = "Total average" if i == 0 else None
        plt.semilogx(curve.n_features, curve.scores, c="k", label=label)

    min_y, max_y = plt.gca().get_ylim()
    for attribute in ["min_feats", "max_feats", "mid_feats"]:
        n_feats = len(getattr(feature_selector.selected_features, attribute))
        plt.vlines(
            n_feats, min_y, max_y, linestyle="--", colors="grey", lw=2, label=attribute,
        )

    plt.xlabel("# features")
    plt.ylabel("Fitness score")
    plt.grid(ls=":")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    return plt.gca()
