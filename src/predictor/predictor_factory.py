from scipy.spatial.distance import cosine
from similarity import spearman, CommonSimilarityService, AdjustedCosineSimilarityService
from scipy.spatial.distance import correlation, cosine
from predictor import KNNItemBasedPredictor, KNNUserBasedPredictor, KNNAdjustedCosineItemBasedPredictor
from util import ProcessPool
from sklearn.neighbors import NearestNeighbors, DistanceMetric
import numpy as np

PREDICTORS = {
    'knn_cosine_user_based': lambda rm, n_neighbors: KNNUserBasedPredictor(rm, cosine, n_neighbors, name='knn_cosine_user_based'),
    'knn_cosine_item_based': lambda rm, n_neighbors: KNNItemBasedPredictor(rm, cosine, n_neighbors, name='knn_cosine_item_based'),
 
    'knn_pearson_user_based': lambda rm, n_neighbors: KNNUserBasedPredictor(rm, correlation, n_neighbors, name='knn_pearson_user_based'),
    'knn_pearson_item_based': lambda rm, n_neighbors: KNNItemBasedPredictor(rm, correlation, n_neighbors, name='knn_pearson_item_based'),
    
    'knn_adj_cosine_item_based': lambda rm, n_neighbors: KNNAdjustedCosineItemBasedPredictor(rm, n_neighbors, name='knn_adj_cosine_item_based'),

    'knn_euclidean_user_based': lambda rm, n_neighbors: KNNUserBasedPredictor(rm, 'euclidean', n_neighbors, name='knn_euclidean_user_based'),
    'knn_euclidean_item_based': lambda rm, n_neighbors: KNNItemBasedPredictor(rm, 'euclidean', n_neighbors, name='knn_euclidean_item_based'),

    'knn_minkowski_user_based': lambda rm, n_neighbors: KNNUserBasedPredictor(rm, 'minkowski', n_neighbors, name='knn_minkowski_user_based'),
    'knn_minkowski_item_based': lambda rm, n_neighbors: KNNItemBasedPredictor(rm, 'minkowski', n_neighbors, name='knn_minkowski_item_based'),
   
    'knn_mahalanobis_user_based': lambda rm, n_neighbors: KNNUserBasedPredictor(rm, 'mahalanobis', n_neighbors, algorithm='auto', metric_params = {'VI': np.cov(rm.data, rowvar=False)}, name='knn_mahalanobis_user_based'),
    'knn_mahalanobis_item_based': lambda rm, n_neighbors: KNNItemBasedPredictor(rm, 'mahalanobis', n_neighbors, algorithm='auto', metric_params = {'VI': np.cov(rm.data, rowvar=False)}, name='knn_mahalanobis_item_based'),

    'knn_chebyshev_user_based': lambda rm, n_neighbors: KNNUserBasedPredictor(rm, 'chebyshev', n_neighbors, name='knn_chebyshev_user_based'),
    'knn_chebyshev_item_based': lambda rm, n_neighbors: KNNItemBasedPredictor(rm, 'chebyshev', n_neighbors, name='knn_chebyshev_item_based'),

    'knn_manhattan_user_based': lambda rm, n_neighbors: KNNUserBasedPredictor(rm, 'manhattan', n_neighbors, name='knn_manhattan_user_based'),
    'knn_manhattan_item_based': lambda rm, n_neighbors: KNNItemBasedPredictor(rm, 'manhattan', n_neighbors, name='knn_manhattan_item_based')   
}

class PredictorFactory:
    NAMES = [ n for n,_ in PREDICTORS.items()]

    def __init__(self, n_processes=None):
        self.__n_processes = n_processes

    def create(self, name, rm, n_neighbors):
        if name not in PREDICTORS:
            raise Error(f'Missing {name} predictor!')
        return PREDICTORS[name](rm, n_neighbors)
 
    def _create_fn(self, name, rm, n_neighbors):
        return self.create(name, rm, n_neighbors)
 
    def create_many(self, names, rm, n_neighbors):
        pool = ProcessPool(self.__n_processes)
        params = [(name, rm, n_neighbors) for name in names]
        return pool.run(self._create_fn, params)

    def create_all(self, rm, n_neighbors):
        return self.create_many(PredictorFactory.NAMES, rm, n_neighbors)
