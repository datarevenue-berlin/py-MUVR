import pytest
import logging
import numpy as np
from py_muvr import utils
from py_muvr.data_structures import FeatureRanks


@pytest.fixture
def scores():
    return [{5: 5, 4: 4, 3: 4, 2: 4}, {5: 10, 4: 9, 3: 10, 2: 11}]


def test_average_scores(scores):
    avg_score = utils.average_scores(scores)
    assert avg_score
    assert avg_score[5] == 7.5
    assert avg_score[4] == 6.5
    assert avg_score[3] == 7
    assert avg_score[2] == 7.5


def test_normalize_score():
    avg_score = {1: 11, 2: 6, 3: 1, 4: 6, 5: 11}
    norm_score = utils.normalize_score(avg_score)
    assert norm_score
    assert norm_score[1] == 1
    assert norm_score[3] == 0
    assert norm_score[2] == 0.5


def test_average_ranks():
    features1 = [1, 2, 3]
    features2 = [0, 1, 4, 5]
    ranks1 = FeatureRanks(features=features1, ranks=[1, 2, 3], n_feats=10)
    ranks2 = FeatureRanks(features=features2, ranks=[1, 2, 3, 4], n_feats=10)
    avg_ranks = utils.average_ranks([ranks1, ranks2])
    assert isinstance(avg_ranks, FeatureRanks)
    assert avg_ranks[0] == 5.5
    assert avg_ranks[1] == 1.5
    assert avg_ranks[2] == 6


@pytest.mark.parametrize(
    "n, best",
    [
        (1, [5]),
        (2, [5, 0]),
        (3, [5, 0, 1]),
        (4, [5, 0, 1, 4]),
    ],
)
def test_get_best_ranks(n, best):
    ranks = FeatureRanks(features=[5, 0, 1, 4], ranks=[1, 2, 3, 4], n_feats=10)
    best_feats = utils.get_best_n_features(ranks, n)
    assert sorted(best_feats) == sorted(best)


def test_compute_t_student_p_value():
    """Using p-values tables for t-student with 17 dof"""
    dof = 17
    n = dof + 1
    random_state = np.random.RandomState(9)
    sample_005 = -1.74
    sample_01 = -1.333
    p_values_005 = []
    p_values_01 = []
    for i in range(1000):
        population = random_state.normal(size=n)
        p_value = utils.compute_t_student_p_value(sample_005, population)
        p_values_005.append(p_value)
        p_value = utils.compute_t_student_p_value(sample_01, population)
        p_values_01.append(p_value)
    assert abs(np.mean(p_values_005) - 0.05) < 0.01
    assert abs(np.mean(p_values_01) - 0.1) < 0.01


def test_mute_loggers():
    @utils.mute_loggers(loggers=["py_muvr.feature_selector"])
    def test_function(logger_name):
        return logging.getLogger(logger_name).getEffectiveLevel()

    @utils.mute_loggers(loggers=["py_muvr.feature_selector"])
    def test_bad_function(logger_name):
        raise RuntimeError("exception")

    assert test_function("py_muvr.models.pls") == logging.INFO
    assert test_function("py_muvr.feature_selector") == logging.WARN
    assert logging.getLogger("py_muvr.models.pls").getEffectiveLevel() == logging.INFO
    assert (
        logging.getLogger("py_muvr.feature_selector").getEffectiveLevel()
        == logging.INFO
    )
    with pytest.raises(RuntimeError):
        test_bad_function("py_muvr.feature_selector")
    assert (
        logging.getLogger("py_muvr.feature_selector").getEffectiveLevel()
        == logging.INFO
    )
