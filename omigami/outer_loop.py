from typing import Union
from concurrent.futures import Executor, Future
from omigami.feature_evaluator import FeatureEvaluator
from omigami.recursive_feature_eliminator import RecursiveFeatureEliminator
from omigami.models import OuterLoopResults


class OuterLoop:
    def __init__(
        self,
        n_outer: int,
        feature_evaluator: FeatureEvaluator,
        dropout_rate: float,
        robust_minimum: float,
        executor: Union[None, Executor],
    ):
        self.feature_evaluator = feature_evaluator
        self.recursive_feature_eliminator = RecursiveFeatureEliminator(
            feature_evaluator, dropout_rate, robust_minimum
        )
        self.n_outer = n_outer
        self._executor = executor

    def run(self):
        results = []
        for outer_loop_idx in range(self.n_outer):
            result = self._execute_loop(outer_loop_idx)
            results.append(result)
        return results

    def _deferred_execute_loop(self, outer_loop_idx: int) -> Future:
        if self._executor is None:
            return self._execute_loop(outer_loop_idx)
        return self._executor.submit(self._execute_loop, outer_loop_idx)

    def _execute_loop(self, outer_loop_idx) -> OuterLoopResults:
        elimination_res = self.recursive_feature_eliminator.run(outer_loop_idx)
        min_feats = elimination_res.best_feats.min_feats
        max_feats = elimination_res.best_feats.max_feats
        mid_feats = elimination_res.best_feats.mid_feats
        min_eval = self.feature_evaluator.evaluate_features(min_feats, outer_loop_idx)
        max_eval = self.feature_evaluator.evaluate_features(max_feats, outer_loop_idx)
        mid_eval = self.feature_evaluator.evaluate_features(mid_feats, outer_loop_idx)
        return OuterLoopResults(
            min_eval=min_eval,
            max_eval=max_eval,
            mid_eval=mid_eval,
            score_vs_feats=elimination_res.score_vs_feats,
        )

    def refresh_splits(self):
        self.feature_evaluator.refresh_splits()
