import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from .evaluator_context import EvaluatorContext
from bunch import Bunch
from util import ProcessPool


def predict_fn(p, batch, decimals):
    return (p, p.predict_batch(batch, decimals))


class Evaluator(ABC):
    def __init__(self, metrics):
        self._metrics = metrics

    def evaluate(self, predictors, rm, decimals=2, n_processes=None):
        ctx = self._predict(predictors, rm, decimals, n_processes)
        return self._perform_metrics(rm, ctx)

    def _predict(self, predictors, rm, decimals, n_processes):
        ctx = EvaluatorContext(rm)
        batch = self._to_batch(rm)

        process_pool = ProcessPool(n_processes)
        results = process_pool.run(predict_fn, [(p, batch, decimals) for p in predictors])

        [ctx.add_pred(p, pred_values) for (p, pred_values) in results]
        ctx.add_true_values(np.array(batch)[:, 2])

        return ctx

    def _perform_metrics(self, rm, ctx):
        data = {}
        for metric in self._metrics:
            for p_name, p_ctx in ctx.predictors.items():
                metric_value = metric.perform(p_ctx.pred_values, ctx.true_values, p_ctx)                
                if p_name not in data:
                    data[p_name] = {}
                data[p_name][metric.name] = metric_value

        return pd.DataFrame([{'Predictor': p, **metrics} for p, metrics in data.items()])

    def _to_batch(self, rm):
        batch = []
        for _, user_id, item_id in rm.cells:
            true_value = rm.cell(user_id, item_id)
            if true_value > 0:
                batch.append((user_id, item_id, true_value))
        return batch