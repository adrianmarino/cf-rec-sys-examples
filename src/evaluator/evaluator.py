import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from .evaluator_context import EvaluatorContext
from bunch import Bunch
from util import ProcessPool
import logging


def predict_fn(p, batch, decimals):
    return (p, p.predict_batch(batch, decimals))


class Evaluator(ABC):
    def __init__(self, metrics):
        self._metrics = metrics

    def evaluate(self, predictors, rm, decimals=2, n_processes=None, transpose=False):
        ctx = self._predict(predictors, rm, decimals, n_processes)
        return self._perform_metrics(rm, ctx, transpose)


    def _predict(self, predictors, rm, decimals, n_processes):
        # batch = [(user_id,row_id, rating)]
        batch = rm.to_batch(lambda v: v>0)

        # outputs= [(predictor, pred_ratings)]
        outputs = self._pred_values(predictors, batch, decimals, n_processes)
        
        return self._create_context(batch, outputs)


    def _create_context(self, batch, outputs):
        ctx = EvaluatorContext(batch)
        [ctx.add_preds(p, pred_values) for (p, pred_values) in outputs]
        return ctx


    def _pred_values(self, predictors, batch, decimals, n_processes):
        process_pool = ProcessPool(n_processes)
        return process_pool.run(predict_fn, [(p, batch, decimals) for p in predictors])


    def _perform_metrics(self, rm, ctx, transpose):
        data = {}
        for metric in self._metrics:
            for p_name, p_pred_values in ctx.pred_values.items():
                logging.info(f'Performing {p_name} {metric.name}...')
                metric_value = metric.perform(
                    p_pred_values,
                    ctx.true_values, 
                    opts=Bunch(predictor_name=p_name, ctx=ctx)
                )
                if p_name not in data:
                    data[p_name] = {}
                data[p_name][metric.name] = metric_value

        result = pd.DataFrame([{'Predictor': p, **metrics} for p, metrics in data.items()])
        return result.set_index('Predictor').T if transpose else result
