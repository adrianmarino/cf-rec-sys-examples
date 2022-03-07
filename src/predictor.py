import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from similarity import k_similar_rows, \
                       k_similar_rows_using_adjusted_cosine
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
from math import sqrt
from similarity import spearman
from scipy.spatial.distance import correlation, cosine


def metrics_fn(predictions, true_values, decimals=3):
    MSE = mean_squared_error(predictions, true_values)
    return { 'rmse': round(sqrt(MSE), decimals),  'mse': round(MSE, decimals) }


def predict(predictors, rm, user_id, item_id):
    return {name: p.predict(rm, user_id, item_id) for name, p in predictors.items()}


def evaluate(predictors, rm, metrics_fn=metrics_fn, decimals=3):
    return {name: p.evaluate(rm, metrics_fn, decimals) for name, p in predictors.items()}


class AbstractPredictor(ABC):
    def __init__(self, n_neighbors, distance=cosine):
        self.distance = distance
        self.n_neighbors = n_neighbors

    @abstractmethod    
    def predict(self, rm, user_id, item_id, decimals=0):
        pass

    def evaluate(self, rm, metrics_fn=metrics_fn, decimals=3):
        predictions = []
        true_values = []

        for _, user_id, item_id in rm.cells:
            true_value = rm.cell(user_id, item_id)
            if true_value > 0:
                predictions.append(self.predict(rm, user_id, item_id))
                true_values.append(true_value)

        return metrics_fn(predictions, true_values, decimals)
        

class UserBasedPredictor(AbstractPredictor):
    def predict(self, rm, user_id, item_id, decimals=0):
        row_sims, row_indices = k_similar_rows(rm, user_id, self.distance, self.n_neighbors)
    
        numerator = denominator = 0
        for curr_row_id, sim in zip(row_indices+1, row_sims):
            r = rm.cell(curr_row_id, item_id)
            if r > 0:
                numerator   += rm.row_deviation(curr_row_id, item_id) * sim
                denominator += sim

        return round(rm.mean_row(user_id) + (numerator / denominator), decimals) if denominator > 0 else 0


class ItemBasedPredictor(AbstractPredictor):
    def predict(self, rm, user_id, item_id, decimals=0):
        t_rm = rm.T()
        row_sims, row_indices = k_similar_rows(t_rm, item_id, self.distance, self.n_neighbors)

        numerator = denominator = 0
        for row_id, sim in zip(row_indices+1, row_sims):
            r = t_rm.cell(row_id, user_id)
            if r > 0:
                numerator   += r * sim
                denominator += sim

        return round(numerator / denominator, decimals) if denominator > 0 else 0


class AdjustedCosineItemBasedPredictor(AbstractPredictor):
    def predict(self, rm, user_id, item_id, decimals=0):
        row_sims, row_indices = k_similar_rows_using_adjusted_cosine(rm, item_id, self.n_neighbors)
    
        numerator = denominator = 0
        for row_id, sim in zip(row_indices+1, row_sims):
            r = rm.cell(user_id, row_id)
            if r > 0:
                numerator   += rm.cell(user_id, row_id) * sim
                denominator += sim
        
        return np.clip(round(numerator / denominator, decimals), 1, 10) if denominator > 0 else 0


class PredictorFactory:
    @staticmethod
    def create(n_neighbors, names=None):
        predictors = {
            'user_based_with_minkowski':  UserBasedPredictor(n_neighbors, 'minkowski'),
            'user_based_with_euclidean':  UserBasedPredictor(n_neighbors, 'euclidean'),
            'user_based_with_cosine':     UserBasedPredictor(n_neighbors, cosine),
            'user_based_with_pearson':    UserBasedPredictor(n_neighbors, correlation),
            'user_based_with_spearman':   UserBasedPredictor(n_neighbors, spearman),
            'user_based_with_manhattan':  UserBasedPredictor(n_neighbors, 'manhattan'),
            'item_based_with_minkowski':  ItemBasedPredictor(n_neighbors, 'minkowski'),
            'item_based_with_euclidean':  ItemBasedPredictor(n_neighbors, 'euclidean'),
            'item_based_with_manhattan':  ItemBasedPredictor(n_neighbors, 'manhattan'),
            'item_based_with_cosine':     ItemBasedPredictor(n_neighbors, cosine),
            'item_based_with_adj_cosine': AdjustedCosineItemBasedPredictor(n_neighbors),
            'item_based_with_pearson':    ItemBasedPredictor(n_neighbors, correlation),
            'item_based_with_spearman':   ItemBasedPredictor(n_neighbors, spearman)
        }
        if names:
            return {n: p for n, p in predictors.items() if n in names}
        else:
            return predictors