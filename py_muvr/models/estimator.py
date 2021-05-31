from abc import ABC, abstractmethod
from py_muvr.data_structures.data_types import RandomState


class Estimator(ABC):
    @property
    @abstractmethod
    def feature_importances(self):
        pass

    @property
    @abstractmethod
    def _estimator_type(self):
        pass

    @abstractmethod
    def set_random_state(self, random_state: RandomState):
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
