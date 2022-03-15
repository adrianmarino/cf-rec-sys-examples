from bunch import Bunch
import numpy as np


class EvaluatorContext:
    def __init__(self, input_batch):
        self.input_batch = np.array(input_batch)
        self.true_values = self.input_batch[:, 2]
        self.user_ids    = self.input_batch[:, 0]
        self.pred_values = Bunch()

    def add_preds(self, predictor, pred_values):
        self.pred_values[predictor.name] = np.array(pred_values)

    def true_pred_by_user(self, user_id, predictor_name):
        indexes = np.where(self.input_batch[:, 0] == user_id)
        
        true_values = self.input_batch[indexes][:, 2]
        pred_values = self.pred_values[predictor_name][indexes]

        return true_values, pred_values
