from metric import AbstractMetric
from sklearn.metrics import precision_score
from util import round_
from random import sample
import logging

class PrecisionMetric(AbstractMetric):
    def __init__(self, k=10, decimals=None, average='macro', rating_desimals=0): 
        super().__init__(f'Precision({average})', decimals)
        self._k = k
        self._average = average
        self._rating_desimals = rating_desimals

    def _calculate(self, pred_values, true_values, opts={}):
        pred_values = [round_(v, self._rating_desimals) for v in pred_values]
        true_values = [round_(v, self._rating_desimals) for v in true_values]

        return precision_score(pred_values, true_values, average=self._average, zero_division=0)
