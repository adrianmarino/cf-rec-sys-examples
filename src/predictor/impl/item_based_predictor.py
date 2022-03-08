from ..predictor import AbstractPredictor
from similarity import CommonSimilarityService


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