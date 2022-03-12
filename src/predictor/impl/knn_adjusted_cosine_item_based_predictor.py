from ..predictor import AbstractPredictor
from similarity import AdjustedCosineSimilarityService
import numpy as np
from util import round_


class KNNAdjustedCosineItemBasedPredictor(AbstractPredictor):
    def __init__(self, rm, n_neighbors, name=None):
        sim_service = AdjustedCosineSimilarityService(rm, n_neighbors)
        super().__init__(rm, sim_service, name)

    def predict(self, user_id, item_id, decimals=None):
        row_sims, row_indices = self.sim_service.similars(item_id)
    
        numerator = denominator = 0
        for row_id, sim in zip(row_indices+1, row_sims):
            r = self.rm.cell(user_id, row_id)
            if r > 0:
                numerator   += self.rm.cell(user_id, row_id) * sim
                denominator += sim
        
        return np.clip(round_(numerator / denominator, decimals), 1, 10) if denominator > 0 else 0
