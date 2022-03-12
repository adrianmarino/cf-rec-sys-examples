from ..predictor import AbstractPredictor
from similarity import CommonSimilarityService
from util import round_


class KNNUserBasedPredictor(AbstractPredictor):
    def __init__(self, rm, distance, n_neighbors, algorithm = 'brute', metric_params=None, name=None):
        sim_service = CommonSimilarityService(rm, distance, n_neighbors, algorithm, metric_params)
        super().__init__(rm, sim_service, name)

    def predict(self, user_id, item_id, decimals=None):
        row_sims, row_indices = self.sim_service.similars(user_id)
    
        numerator = denominator = 0
        for curr_row_id, sim in zip(row_indices+1, row_sims):
            r = self.rm.cell(curr_row_id, item_id)
            if r > 0:
                numerator   += self.rm.row_deviation(curr_row_id, item_id) * sim
                denominator += sim

        return round_(self.rm.mean_row(user_id) + (numerator / denominator), decimals) if denominator > 0 else 0