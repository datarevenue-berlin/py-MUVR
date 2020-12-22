from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.model_selection import GroupKFold

from omigami.model_trainer import TrainingTestingResult, ModelTrainer
from omigami.utils import NumpyArray

SEED = 1


@dataclass
class InnerLoopResult:
    train_results: List[TrainingTestingResult]
    features: List[int]

    def __len__(self):
        return len(self.train_results)

    def __getitem__(self, item_idx):
        return self.train_results[item_idx]

    def __iter__(self):
        for res in self.train_results:
            yield res

    @property
    def average_score(self) -> float:
        return np.average([r.score for r in self.train_results])


class InnerLooper:
    """This class perform the recursive feature elimination based on the inner loop
    cross validation fold"""

    def __init__(self, n_inner: int, groups: NumpyArray):
        self.seed = SEED
        self.n_inner = n_inner
        self.groups = groups

    def run(
        self,
        X: NumpyArray,
        y: NumpyArray,
        model_trainer: ModelTrainer,
        features: List[int],
    ) -> InnerLoopResult:
        """Perform inner loop cross validation using all the inner loop splits"""
        inner_splits = self._make_inner_splits(X, y)
        inner_loop_results = []

        for inner_index, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
            inner_fold_results = model_trainer.evaluate_features(
                X, y, inner_train_idx, inner_val_idx, features
            )
            inner_loop_results.append(inner_fold_results)

        return InnerLoopResult(inner_loop_results, features=features)

    def _make_inner_splits(self, X: NumpyArray, y: NumpyArray):
        inner_splitter = GroupKFold(self.n_inner)
        inner_splits = inner_splitter.split(X, y, self.groups)

        return inner_splits
