from metric import AbstractMetric
from sklearn.metrics import precision_score
from ..pred_true_values_sampler import PredTrueValuesSampler

class PrecisionKMetric(AbstractMetric):
    def __init__(self, k=10, decimals=None, average='macro', rating_desimals=0):
        name = f'Precision@{k}'
        if average != 'macro':
            name += f'({average})'

        super().__init__(name, decimals)

        self._average = average
        self._rating_desimals = rating_desimals
        self._sampler = PredTrueValuesSampler(k, rating_desimals)

    def _calculate(self, pred_values, true_values, opts={}):
        pred_values, true_values = self._sampler.sample(pred_values, true_values)
        return precision_score(pred_values, true_values, average=self._average, zero_division=0)



