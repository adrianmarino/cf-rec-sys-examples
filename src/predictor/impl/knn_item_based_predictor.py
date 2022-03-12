from ..predictor import AbstractPredictor
from similarity import CommonSimilarityService
from util import round_


class KNNItemBasedPredictor(AbstractPredictor):
    def __init__(self, rm, distance, n_neighbors, algorithm = 'brute', metric_params=None, name=None):
        t_rm = rm.T()
        sim_service = CommonSimilarityService(t_rm, distance, n_neighbors, algorithm, metric_params)
        super().__init__(t_rm, sim_service, name)
 
    def predict(self, user_id, item_id, decimals=None):
        row_sims, row_indices = self.sim_service.similars(user_id)

        numerator = denominator = 0
        for row_id, sim in zip(row_indices+1, row_sims):
            r = self.rm.cell(row_id, user_id)
            if r > 0:
                numerator   += r * sim
                denominator += sim

        return round_(numerator / denominator, decimals) if denominator > 0 else 0