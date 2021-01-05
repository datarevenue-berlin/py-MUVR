import pytest
from omigami import utils
from omigami.data_models import FeatureRanks


@pytest.fixture
def scores():
    return [{5: 5, 4: 4, 3: 4, 2: 4}, {5: 10, 4: 9, 3: 10, 2: 11}]


def test_compute_number_of_features(scores):
    n_feats = utils.compute_number_of_features(scores, 0.05)
    assert isinstance(n_feats, dict)
    assert utils.MIN in n_feats
    assert utils.MAX in n_feats
    assert utils.MID in n_feats
    assert n_feats[utils.MIN] == 4
    assert n_feats[utils.MAX] == 4
    assert n_feats[utils.MID] == 4


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
    "n, best", [(1, [5]), (2, [5, 0]), (3, [5, 0, 1]), (4, [5, 0, 1, 4]),]
)
def test_get_best_ranks(n, best):
    ranks = FeatureRanks(features=[5, 0, 1, 4], ranks=[1, 2, 3, 4], n_feats=10)
    best_feats = utils.get_best_n_features(ranks, n)
    assert sorted(best_feats) == sorted(best)
