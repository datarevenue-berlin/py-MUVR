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
