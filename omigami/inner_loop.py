from typing import Iterable, List
from omigami.feature_evaluator import FeatureEvaluator, FeatureEvaluationResults

InnerLoopResults = List[FeatureEvaluationResults]


class InnerLoop:
    def __init__(self, feature_evaluator: FeatureEvaluator):
        self._n_inner = feature_evaluator.get_inner_loop_size()
        self.feature_evaluator = feature_evaluator

    def run(self, features: Iterable[int], out_idx: int) -> InnerLoopResults:
        results = []
        for in_idx in range(self._n_inner):
            res = self.feature_evaluator.evaluate_features(features, out_idx, in_idx)
            results.append(res)
        return results
