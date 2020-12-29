import pytest
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from omigami.estimator import ModelTrainer


@pytest.fixture
def dataset():
    return datasets.make_classification(n_samples=200, n_features=5, random_state=42)


@pytest.fixture
def model_trainer():
    return ModelTrainer(estimator="RFC", random_state=42,)


def test_make_estimator(model_trainer):
    estimator = model_trainer._make_estimator("RFC")
    assert isinstance(estimator, RandomForestClassifier)
    svc = SVC()
    estimator = model_trainer._make_estimator(svc)
    assert svc is estimator
    with pytest.raises(ValueError):
        model_trainer._make_estimator("a_whiter_shade_of_estimator")


def test_train_model(model_trainer, dataset):
    X, y = dataset
    estimator = model_trainer.train_model(X, y)
    assert estimator
    check_is_fitted(estimator)
