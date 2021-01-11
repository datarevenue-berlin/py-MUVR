from typing import Callable, TypeVar, Union
from numpy import ndarray
from numpy.random import RandomState as RS
from sklearn.base import BaseEstimator


NumpyArray = ndarray
MetricFunction = Callable[[NumpyArray, NumpyArray], float]
GenericEstimator = TypeVar("GenericEstimator")
InputEstimator = Union[BaseEstimator, GenericEstimator]
RandomState = type(RS)
