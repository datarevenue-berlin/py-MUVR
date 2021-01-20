import logging
from typing import Iterable
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.ticker as mtick
import pandas as pd
from omigami.feature_selector import FeatureSelector

log = logging.getLogger(__name__)


def plot_validation_curves(feature_selector: FeatureSelector, **figure_kwargs) -> Axes:
    if not feature_selector.is_fit:
        log.warning(  # pylint: disable=logging-not-lazy
            "Validation curves have not been generated. To be able to plot"
            + " call `select_features` method first"
        )
        return None
    curves = feature_selector.get_validation_curves()
    plt.figure(**figure_kwargs)
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
    for attribute in ["min", "max", "mid"]:
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
    return plt.gcf()


def plot_feature_rank(
    feature_selector: FeatureSelector,
    model: str,
    feature_names: Iterable = None,
    show_outliers: bool = True,
    **figure_kwargs
):
    if model not in {"min", "max", "mid"}:
        raise ValueError("The model parameter must be one of 'min', 'max' or 'mid'.")

    eval_attr = model + "_eval"
    feats_attr = model

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

    if "figsize" not in figure_kwargs.keys():
        fig_width = len(selected_ranks.columns) / 3
        figure_kwargs["figsize"] = (6, max(fig_width, 5))

    fig, (ax_ranks, ax_notnan) = plt.subplots(
        nrows=1, ncols=2, sharey=True, **figure_kwargs
    )

    color_notnan = "grey"
    color_ranks = "#4e79a7"

    ax_notnan.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax_notnan.set_ylabel("Feature")
    ax_notnan.set_xlabel("Percentage of times selected")
    ax_ranks.set_xlabel("Feature Rank")

    ax_notnan.tick_params(axis="x")
    ax_ranks.tick_params(axis="x")

    bbox_props = {"color": color_ranks}
    bbox_color = {"boxes": color_ranks, "medians": "black"}

    if feature_names is not None:
        feature_numbers = range(len(feature_names))
        numbers_to_names = dict(zip(feature_numbers, feature_names))
        selected_ranks.rename(columns=numbers_to_names, inplace=True)

    selected_ranks.boxplot(
        positions=range(len(selected_ranks.columns)),
        color=bbox_color,
        patch_artist=True,
        ax=ax_ranks,
        boxprops=bbox_props,
        vert=False,
        showfliers=show_outliers,
    )

    (selected_ranks.notna().mean() * 100).plot.barh(
        facecolor=color_notnan, ax=ax_notnan, edgecolor="black"
    )

    ax_notnan.invert_yaxis()  # being the y-axis, shared it will invert both
    ax_notnan.grid(axis="x", zorder=-100)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    return fig
