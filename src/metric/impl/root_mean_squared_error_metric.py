from metric import AbstractMetric
from sklearn.metrics import mean_squared_error
from math import sqrt


class RootMeanSquaredErrorMetric(AbstractMetric):
    def __init__(self, decimals=None): super().__init__('RMSE', decimals)

    def _calculate(self, pred_values, true_values, opts={}):
        return sqrt(mean_squared_error(pred_values, true_values))
