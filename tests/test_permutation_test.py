import pytest
from sklearn import datasets
from omigami import permutation_test
from omigami.feature_selector import FeatureSelector
from omigami.utils import compute_t_student_p_value
from omigami.data_structures import (
    SelectedFeatures,
    ScoreCurve,
    FeatureSelectionResults,
)


@pytest.fixture
def fit_permutation_test():

    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="PLSC",)
    pt = permutation_test.PermutationTest(feature_selector=fs, n_permutations=4)

    pt.res = FeatureSelectionResults(
        SelectedFeatures(min=[0, 1], max=[0, 1, 4, 3], mid=[0, 1, 4]),
        ScoreCurve(n_features=[2, 3, 4, 5], scores=[4.0, 4.5, 4.0, 4.75]),
    )
    pt.res_perm = [
        FeatureSelectionResults(
            SelectedFeatures(min=[0], max=[0, 1, 2], mid=[0, 1]),
            ScoreCurve(n_features=[1, 2, 3], scores=[4.0, 4.5, 4.5]),
        ),
        FeatureSelectionResults(
            SelectedFeatures(min=[0], max=[0, 1, 2], mid=[0, 1]),
            ScoreCurve(n_features=[1, 2, 3], scores=[4.0, 4.5, 4.5]),
        ),
        FeatureSelectionResults(
            SelectedFeatures(min=[0], max=[0, 1, 2], mid=[0, 1]),
            ScoreCurve(n_features=[1, 2, 3], scores=[4.0, 4.6, 4.9]),
        ),
        FeatureSelectionResults(
            SelectedFeatures(min=[0], max=[0, 1, 2], mid=[0, 1]),
            ScoreCurve(n_features=[1, 2, 3], scores=[4.1, 4.6, 4.9]),
        ),
    ]
    return pt


@pytest.fixture
def fit_feature_selector(results):
    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="PLSC",)
    fs.is_fit = True
    fs.results = results
    sel_feats = fs._post_processor.select_features(results)
    fs._selected_features = sel_feats
    return fs


def test_permutation_test():
    fs = FeatureSelector(
        n_outer=8, n_repetitions=8, random_state=0, estimator="PLSC", metric="MISS"
    )
    assert permutation_test.PermutationTest(feature_selector=fs, n_permutations=10)


def test_fit(fit_feature_selector):
    n_permutations = 2
    pt = permutation_test.PermutationTest(
        feature_selector=fit_feature_selector, n_permutations=n_permutations
    )

    X, y = datasets.make_classification(n_features=5, n_samples=5, random_state=0)
    pt.fit(X, y)

    assert pt.res
    assert pt.res_perm
    assert len(pt.res_perm) == n_permutations


def test_compute_permutation_scores(fit_permutation_test):
    s_min, p_score_min = fit_permutation_test.compute_permutation_scores(model="min")
    s_mid, p_score_mid = fit_permutation_test.compute_permutation_scores(model="mid")
    s_max, p_score_max = fit_permutation_test.compute_permutation_scores(model="max")

    assert s_min == 4
    assert sorted(p_score_min) == [4.0, 4.0, 4.0, 4.1]
    assert s_mid == 4.5
    assert sorted(p_score_mid) == [4.5, 4.5, 4.6, 4.6]
    assert s_max == 4
    assert sorted(p_score_max) == [4.5, 4.5, 4.9, 4.9]


def test_compute_p_values(fit_permutation_test):
    min_p_val = fit_permutation_test.compute_p_values(model="min")
    mid_p_val = fit_permutation_test.compute_p_values(model="mid")
    max_p_val = fit_permutation_test.compute_p_values(model="max")

    assert min_p_val == compute_t_student_p_value(4, [4.0, 4.0, 4.0, 4.1])
    assert mid_p_val == compute_t_student_p_value(4.5, [4.5, 4.5, 4.6, 4.6])
    assert max_p_val == compute_t_student_p_value(4, [4.5, 4.5, 4.9, 4.9])


def test_rank_data(fit_permutation_test):
    rank_sample, ranks_population = fit_permutation_test._rank_data(10, [1, 2, 40])
    assert rank_sample == 3
    assert ranks_population[0] == 1
    assert ranks_population[1] == 2
    assert ranks_population[2] == 4
