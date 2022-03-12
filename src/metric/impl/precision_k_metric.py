from metric import AbstractMetric
from sklearn.metrics import precision_score
from util import round_
from random import sample
import logging

class PrecisionKMetric(AbstractMetric):
    def __init__(self, k=10, decimals=None, average='macro', rating_desimals=0): 
        super().__init__(f'Precision@{k}({average})', decimals)
        self._k = k
        self._average = average
        self._rating_desimals = rating_desimals

    def _calculate(self, pred_values, true_values, opts={}):
        pred_values, true_values = self._sample(pred_values, true_values)
        return precision_score(pred_values, true_values, average=self._average, zero_division=0)

    def _sample(self, pred_values, true_values):
        indexes_sample = self._indexes_sample(len(pred_values))
       
        pred_values = [round_(pred_values[i], self._rating_desimals) for i in indexes_sample]
        true_values = [round_(true_values[i], self._rating_desimals) for i in indexes_sample]
        return pred_values, true_values

    def _indexes_sample(self, values_size):
        values_range = list(range(0, values_size))
 
        if self._k > values_size:
            logging.warn(f'K={self._k} is greater than values size ({values_size})!. User values size as K.')
            return sample(values_range, values_size)
       
        return sample(values_range, self._k)
