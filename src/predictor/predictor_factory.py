from scipy.spatial.distance import cosine
from similarity import spearman, CommonSimilarityService, AdjustedCosineSimilarityService
from scipy.spatial.distance import correlation, cosine
from predictor import ItemBasedPredictor, UserBasedPredictor, AdjustedCosineItemBasedPredictor


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