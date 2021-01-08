import logging
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from omigami.feature_selector import FeatureSelector


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
    selected_features = feature_selector.get_selected_features()
    for attribute in ["min_feats", "max_feats", "mid_feats"]:
        n_feats = len(getattr(selected_features, attribute))
        plt.vlines(
            n_feats,
            min_y,
            max_y,
            linestyle="--",
            colors="grey",
            lw=2,
            label=attribute,
            zorder=100000,
        )

    plt.xlabel("# features")
    plt.ylabel("Fitness score")
    plt.grid(ls=":")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    return plt.gca()


def plot_feature_rank(feature_selector, model, feature_names=None):
    if model not in {"min", "max", "mid"}:
        raise ValueError("model must be one of min, max or mid")

    eval_attr = model + "_eval"
    feats_attr = model + "_feats"

    ranks = []
    for r in feature_selector.results:
        for ol in r:
            ranks_raw_data = getattr(ol, eval_attr).ranks.get_data()
            ranks.append(ranks_raw_data)

    selected_features = feature_selector.get_selected_features()
    best = getattr(selected_features, feats_attr)
    selected_ranks = pd.DataFrame(r for r in ranks)[best]

    sorted_feats = selected_ranks.mean().sort_values().index
    selected_ranks = selected_ranks[sorted_feats]

    fig, ax_notnan = plt.subplots()
    ax_ranks = (
        ax_notnan.twinx()
    )  # instantiate a second axes that shares the same x-axis

    color_notnan = "grey"
    color_ranks = "#4e79a7"

    ax_notnan.set_xlabel("feature")
    ax_notnan.set_ylabel("not-NaN fraction", color=color_notnan)
    ax_ranks.set_ylabel("rank", color=color_ranks)

    ax_notnan.tick_params(axis="y", labelcolor=color_notnan)
    ax_ranks.tick_params(axis="y", labelcolor=color_ranks)

    bbox_props = {"color": color_ranks, "alpha": 0.5}
    bbox_color = {"boxes": color_ranks, "medians": "black"}

    fig_width = len(selected_ranks.columns) / 3
    figsize = (max(fig_width, 5), 3)

    if feature_names is not None:
        feature_numbers = range(len(feature_names))
        numbers_to_names = dict(zip(feature_numbers, feature_names))
        selected_ranks.rename(columns=numbers_to_names, inplace=True)

    selected_ranks.notna().mean().plot.bar(
        figsize=figsize, facecolor=color_notnan, ax=ax_notnan, alpha=0.7
    )

    selected_ranks.boxplot(
        positions=range(len(selected_ranks.columns)),
        color=bbox_color,
        patch_artist=True,
        ax=ax_ranks,
        boxprops=bbox_props,
    )

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    return fig
