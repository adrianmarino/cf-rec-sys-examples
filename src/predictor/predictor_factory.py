from scipy.spatial.distance import cosine
from similarity import spearman, CommonSimilarityService, AdjustedCosineSimilarityService
from scipy.spatial.distance import correlation, cosine
from predictor import ItemBasedPredictor, UserBasedPredictor, AdjustedCosineItemBasedPredictor
from util import ProcessPool


PREDICTORS = { 
    'user_based_with_cosine': lambda rm, n_neighbors: UserBasedPredictor(rm, cosine, n_neighbors),
    'user_based_with_pearson': lambda rm, n_neighbors: UserBasedPredictor(rm, correlation, n_neighbors),
    'item_based_with_cosine': lambda rm, n_neighbors: ItemBasedPredictor(rm, cosine, n_neighbors),
    'item_based_with_pearson': lambda rm, n_neighbors: ItemBasedPredictor(rm, correlation, n_neighbors),
    'item_based_with_adj_cosine': lambda rm, n_neighbors: AdjustedCosineItemBasedPredictor(rm, n_neighbors)
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
        return (name, self.create(name, rm, n_neighbors))
 
    def create_many(self, names, rm, n_neighbors):
        pool = ProcessPool(self.__n_processes)
        params = [(name, rm, n_neighbors) for name in names]
        return {n: p for n, p in pool.run(self._create_fn, params)}

    def create_all(self, rm, n_neighbors):
        return self.create_many(self.NAMES, rm, n_neighbors)
