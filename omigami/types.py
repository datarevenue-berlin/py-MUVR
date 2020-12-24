from typing import Callable, Tuple, TypeVar, Union
from numpy import ndarray
from numpy import random
from sklearn.base import BaseEstimator


NumpyArray = ndarray
MetricFunction = Callable[[NumpyArray, NumpyArray], float]
Split = Tuple[NumpyArray, NumpyArray]
GenericEstimator = TypeVar("GenericEstimator")
Estimator = Union[BaseEstimator, GenericEstimator]
RandomState = type(random.RandomState)
