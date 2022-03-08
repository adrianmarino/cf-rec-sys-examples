import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
from math import sqrt
from util import ProcessPool


def metrics_fn(predictions, true_values, decimals=3):
    MSE = mean_squared_error(predictions, true_values)
    return { 'rmse': round(sqrt(MSE), decimals),  'mse': round(MSE, decimals) }


def __predict_fn(n, p, user_id, item_id): 
    return (n, p.predict(user_id, item_id))


def __evaluate_fn(n, p, rm, metrics_fn, decimals): 
    return (n, p.evaluate(rm, metrics_fn, decimals))


def predict(predictors, user_id, item_id, n_processes=24):
    params  = [(n, p, user_id, item_id) for n, p in predictors.items()]
    return {n: r for n, r in ProcessPool(n_processes).run(__predict_fn, params)}


def evaluate(predictors, rm, metrics_fn=metrics_fn, decimals=3, n_processes=24):
    params = [(n, p, rm, metrics_fn, decimals) for n, p  in predictors.items()]
    return {n: r for n, r in ProcessPool(n_processes).run(__evaluate_fn, params)}


class AbstractPredictor(ABC):
    def __init__(self, rm, sim_service):
        self.sim_service = sim_service
        self.rm = rm

    @abstractmethod    
    def predict(self, user_id, item_id, decimals=0):
        pass

    def evaluate(self, rm, metrics_fn=metrics_fn, decimals=3):
        predictions = []
        true_values = []

        for _, user_id, item_id in rm.cells:
            true_value = rm.cell(user_id, item_id)
            if true_value > 0:
                predictions.append(self.predict(user_id, item_id, decimals))
                true_values.append(true_value)

        return metrics_fn(predictions, true_values, decimals)
