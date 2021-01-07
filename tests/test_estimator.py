import pytest
from sklearn import datasets
from models.estimator import ModelTrainer


@pytest.fixture
def dataset():
    return datasets.make_classification(n_samples=200, n_features=5, random_state=42)


@pytest.fixture
def model_trainer():
    return ModelTrainer(estimator="RFC", random_state=42,)
