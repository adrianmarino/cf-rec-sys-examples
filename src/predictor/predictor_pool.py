from .predictor import metrics_fn
from util import ProcessPool


def predict_fn(n, p, user_id, item_id): 
    return (n, p.predict(user_id, item_id))


def evaluate_fn(n, p, rm, metrics_fn, decimals): 
    return (n, p.evaluate(rm, metrics_fn, decimals))


class PredictorPool:
    def __init__(self, predictors):
        self.__predictors = predictors

    def predict(self, user_id, item_id, n_processes=None):
        params  = [(n, p, user_id, item_id) for n, p in self.__predictors.items()]
        return {n: r for n, r in ProcessPool(n_processes).run(predict_fn, params)}

    def evaluate(self, rm, metrics_fn=metrics_fn, decimals=3, n_processes=None):
        params = [(n, p, rm, metrics_fn, decimals) for n, p  in self.__predictors.items()]
        return {n: r for n, r in ProcessPool(n_processes).run(evaluate_fn, params)}
