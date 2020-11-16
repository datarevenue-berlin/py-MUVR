import logging
from matplotlib import pyplot as plt
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
