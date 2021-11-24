import pytest

from sklearn.datasets import make_classification
from sklearn.svm import SVC

from py_muvr.feature_selector import FeatureSelector


@pytest.fixture()
def artificial_data():
    return make_classification(
        n_features=10,
        n_informative=2,
        n_redundant=2,
        n_samples=100,
        n_classes=2,
        random_state=0,
        shuffle=False,
    )


@pytest.mark.parametrize(
    "estimator", ["RFC", "XGBC", "PLSC", SVC(kernel="linear", random_state=1)]
)
def test_feature_selection_artificial_data_plsc(artificial_data, estimator):

    fs = FeatureSelector(
        n_outer=5,
        n_repetitions=5,
        random_state=0,
        estimator=estimator,
        metric="accuracy",
        features_dropout_rate=0.05,
    )

    X, y = artificial_data
    fitted_fs = fs.fit(X, y)
    selected_features = fitted_fs._selected_features

    assert selected_features["min"] == [0, 1]  # first two features
    assert selected_features["max"] == [0, 3]  # thirds and fourth feature
