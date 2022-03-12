from metric import AbstractMetric
from sklearn.metrics import mean_squared_error


class MeanSquaredErrorMetric(AbstractMetric):
    def __init__(self, decimals=None): super().__init__('MSE', decimals)

    def _calculate(self, pred_values, true_values, opts={}):
        return mean_squared_error(pred_values, true_values)
