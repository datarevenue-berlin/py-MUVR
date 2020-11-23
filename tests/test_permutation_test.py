from sklearn import datasets
from omigami import permutation_test
from omigami.omigami import FeatureSelector
from omigami.utils import compute_t_student_p_value


def test_permutation_test(fitted_feature_selector):
    assert permutation_test.PermutationTest(
        feature_selector=fitted_feature_selector, n_permutations=10
    )


def test_get_avg_score(fitted_feature_selector):
    pt = permutation_test.PermutationTest(
        feature_selector=fitted_feature_selector, n_permutations=10
    )
    avg_score = pt._get_avg_score("min")
    assert avg_score == 4


def test_get_avg_scores(fitted_feature_selector):
    pt = permutation_test.PermutationTest(
        feature_selector=fitted_feature_selector, n_permutations=10
    )
    avg_scores = pt._get_avg_scores()
    assert avg_scores["min"] == 4
    assert avg_scores["max"] == 4


def test_fit(monkeypatch, results):
    def fit_mock(*args, **kwargs):
        pass

    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="RFC",)
    fs.n_features = 10
    fs.is_fit = True
    fs.selected_features = {"min": (0, 1), "max": (0, 1), "mid": (0, 1)}
    fs._results = results
    sel_feats = fs._process_results(results)
    fs._selected_features = sel_feats

    monkeypatch.setattr(fs, "fit", fit_mock)
    n_permutations = 10
    pt = permutation_test.PermutationTest(
        feature_selector=fs, n_permutations=n_permutations
    )

    X, y = datasets.make_classification(n_features=5, random_state=0)
    pt.fit(X, y)

    assert pt.res
    assert pt.res_perm
    assert len(pt.res_perm) == n_permutations


def test_compute_p_values(monkeypatch, results):
    def fit_mock(*args, **kwargs):
        pass

    fs = FeatureSelector(n_outer=5, metric="MISS", estimator="RFC",)
    fs.n_features = 10
    fs.is_fit = True
    fs.selected_features = {"min": (0, 1), "max": (0, 1), "mid": (0, 1)}
    fs._results = results
    sel_feats = fs._process_results(results)
    fs._selected_features = sel_feats

    monkeypatch.setattr(fs, "fit", fit_mock)
    n_permutations = 10
    pt = permutation_test.PermutationTest(
        feature_selector=fs, n_permutations=n_permutations
    )

    X, y = datasets.make_classification(n_features=5, random_state=0)
    pt.fit(X, y)
    import logging

    pt.res_perm[0]["min"] = 4.1
    pt.res_perm[0]["max"] = 4.1
    p_values = pt.compute_p_values()
    assert p_values
    assert p_values["min"]
    assert p_values["max"]
    assert p_values["min"] == compute_t_student_p_value(
        4, (n_permutations - 1) * [4] + [4.1]
    )


def test_rank_data(fitted_feature_selector):
    pt = permutation_test.PermutationTest(
        feature_selector=fitted_feature_selector, n_permutations=10
    )
    rank_sample, ranks_population = pt._rank_data(10, [1, 2, 40])
    assert rank_sample == 3
    assert ranks_population[0] == 1
    assert ranks_population[1] == 2
    assert ranks_population[2] == 4
