from ..predictor import AbstractPredictor
from similarity import CommonSimilarityService


class KNNUserBasedPredictor(AbstractPredictor):
    def __init__(self, rm, distance, n_neighbors, algorithm = 'brute', metric_params=None):
        super().__init__(rm, CommonSimilarityService(rm, distance, n_neighbors, algorithm, metric_params))

    def predict(self, user_id, item_id, decimals=0):
        row_sims, row_indices = self.sim_service.similars(user_id)
    
        numerator = denominator = 0
        for curr_row_id, sim in zip(row_indices+1, row_sims):
            r = self.rm.cell(curr_row_id, item_id)
            if r > 0:
                numerator   += self.rm.row_deviation(curr_row_id, item_id) * sim
                denominator += sim

        return round(self.rm.mean_row(user_id) + (numerator / denominator), decimals) if denominator > 0 else 0