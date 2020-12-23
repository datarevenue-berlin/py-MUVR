from omigami.feature_selector import FeatureSelector


def test_feature_selector():
    fs = FeatureSelector(
        n_outer=8,
        metric="MISS",
        estimator="RFC",
        features_dropout_rate=0.05,
        robust_minimum=0.05,
        repetitions=8,
        random_state=0,
    )
    assert fs
