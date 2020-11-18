import logging
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from omigami.omigami import FeatureSelector
from omigami.utils import average_scores, MIN, MAX, MID


def plot_validation_curves(feature_selector: FeatureSelector) -> Axes:
    """Plot validation curved using feature elimination results. The function
    will plot the relationship between finess score and number of variables
    for each outer loop iteration, their average aggregation (which gives the
    values for the single repetitions) and the average across repetitions.
    The size of the "min", "max" and "mid" sets are plotted as vertical dashed
    lines.
    """
    if not feature_selector.is_fit:
        logging.warning(  # pylint: disable=logging-not-lazy
            "Validation curves have not been generated. To be able to plot"
            + " call `select_features` method first"
        )
        return None
    all_scores = []
    for j, res in enumerate(feature_selector.outer_loop_aggregation):
        for i, score in enumerate(res["scores"]):
            label = "Outer loop average" if i + j == 0 else None
            sorted_score_items = sorted(score.items())
            n_feats, score_values = zip(*sorted_score_items)
            all_scores += score_values
            plt.semilogx(n_feats, score_values, c="#deebf7", label=label)
    max_score = max(all_scores)
    min_score = min(all_scores)
    repetition_averages = []
    for i, r in enumerate(feature_selector.outer_loop_aggregation):
        label = "Repetition average" if i == 0 else None
        avg_scores = average_scores(r["scores"])
        sorted_score_items = sorted(avg_scores.items())
        n_feats, score_values = zip(*sorted_score_items)
        plt.semilogx(n_feats, score_values, c="#3182bd", label=label)
        repetition_averages.append(avg_scores)
    final_avg = average_scores(repetition_averages)
    sorted_score_items = sorted(final_avg.items())
    n_feats, score_values = zip(*sorted_score_items)
    plt.semilogx(n_feats, score_values, c="k", lw=3, label="Final average")
    for key in (MIN, MAX, MID):
        n_feats = len(feature_selector.selected_features[key])
        plt.vlines(
            n_feats,
            min_score,
            max_score,
            linestyle="--",
            colors="grey",
            lw=2,
            label=key,
        )
    plt.xlabel("# features")
    plt.ylabel("Fitness score")
    plt.grid(ls=":")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    return plt.gca()


def plot_feature_rank(feature_selector, model_key):

    ranks = []
    for r in feature_selector.results:
        for ol in r:
            ranks.append(ol.test_results[model_key].feature_ranks.to_dict())
    selected_ranks = pd.DataFrame(r for r in ranks)[
        feature_selector.selected_features[model_key]
    ]

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
