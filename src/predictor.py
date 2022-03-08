import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from math import sqrt
from similarity import spearman, CommonSimilarityService, AdjustedCosineSimilarityService
from scipy.spatial.distance import correlation, cosine
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
        

class UserBasedPredictor(AbstractPredictor):
    def __init__(self, rm, distance, n_neighbors):
        super().__init__(rm, CommonSimilarityService(rm, distance, n_neighbors))

    def predict(self, user_id, item_id, decimals=0):
        row_sims, row_indices = self.sim_service.similars(user_id)
    
        numerator = denominator = 0
        for curr_row_id, sim in zip(row_indices+1, row_sims):
            r = self.rm.cell(curr_row_id, item_id)
            if r > 0:
                numerator   += self.rm.row_deviation(curr_row_id, item_id) * sim
                denominator += sim

        return round(self.rm.mean_row(user_id) + (numerator / denominator), decimals) if denominator > 0 else 0


class ItemBasedPredictor(AbstractPredictor):
    def __init__(self, rm, distance, n_neighbors):
        t_rm = rm.T()
        super().__init__(t_rm, CommonSimilarityService(t_rm, distance, n_neighbors))
 
    def predict(self, user_id, item_id, decimals=0):
        row_sims, row_indices = self.sim_service.similars(user_id)

        numerator = denominator = 0
        for row_id, sim in zip(row_indices+1, row_sims):
            r = self.rm.cell(row_id, user_id)
            if r > 0:
                numerator   += r * sim
                denominator += sim

        return round(numerator / denominator, decimals) if denominator > 0 else 0


class AdjustedCosineItemBasedPredictor(AbstractPredictor):
    def __init__(self, rm, n_neighbors):
        super().__init__(rm, AdjustedCosineSimilarityService(rm, n_neighbors))

    def predict(self, user_id, item_id, decimals=0):
        row_sims, row_indices = self.sim_service.similars(item_id)
    
        numerator = denominator = 0
        for row_id, sim in zip(row_indices+1, row_sims):
            r = self.rm.cell(user_id, row_id)
            if r > 0:
                numerator   += self.rm.cell(user_id, row_id) * sim
                denominator += sim
        
        return np.clip(round(numerator / denominator, decimals), 1, 10) if denominator > 0 else 0



class PredictorFactory:
    def create_user_based_cosine(rm, n_neighbors): 
        return UserBasedPredictor(rm, cosine, n_neighbors)

    def create_user_based_with_pearson(rm, n_neighbors): 
        return UserBasedPredictor(rm, correlation, n_neighbors)
    
    def create_item_based_with_cosine(rm, n_neighbors):
        return ItemBasedPredictor(rm, cosine, n_neighbors)

    def create_item_based_with_adj_cosine(rm, n_neighbors):
        return AdjustedCosineItemBasedPredictor(rm, n_neighbors)

    def create_item_based_with_pearson(rm, n_neighbors):
        return ItemBasedPredictor(rm, correlation, n_neighbors)

    @classmethod
    def create(cls, rm, n_neighbors, names=None):
        predictors = {
            'user_based_with_cosine':     cls.create_user_based_cosine(rm, n_neighbors),
            'user_based_with_pearson':    cls.create_user_based_with_pearson(rm, n_neighbors),
            'item_based_with_cosine':     cls.create_item_based_with_cosine(rm, n_neighbors),
            'item_based_with_adj_cosine': cls.create_item_based_with_adj_cosine(rm, n_neighbors),
            'item_based_with_pearson':    cls.create_item_based_with_pearson(rm, n_neighbors)
        }
        if names:
            return {n: p for n, p in predictors.items() if n in names}
        else:
            return predictors