from .predictor import metrics_fn
from util import ProcessPool
import pandas as pd

def predict_fn(n, p, user_id, item_id, decimals): 
    return (n, p.predict(user_id, item_id, decimals))


def evaluate_fn(n, p, rm, metrics_fn, decimals): 
    return (n, p.evaluate(rm, metrics_fn, decimals))


class PredictorPool:
    def __init__(self, predictors):
        self.__predictors = predictors

    def predict(self, user_id, item_id, n_processes=None, decimals=0):
        params  = [(n, p, user_id, item_id, decimals) for n, p in self.__predictors.items()]
        return {n: r for n, r in ProcessPool(n_processes).run(predict_fn, params)}

    def evaluate(self, rm, metrics_fn=metrics_fn, decimals=3, n_processes=None):
        params = [(n, p, rm, metrics_fn, decimals) for n, p  in self.__predictors.items()]
        results = {n: r for n, r in ProcessPool(n_processes).run(evaluate_fn, params)}
        return pd.DataFrame(data=[{'Predictor': predictor_name, **metrics} for predictor_name, metrics in results.items()])
