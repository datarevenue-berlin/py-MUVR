from dataclasses import dataclass
from omigami.types import NumpyArray


@dataclass
class InputData:
    X: NumpyArray
    y: NumpyArray
    groups: NumpyArray
