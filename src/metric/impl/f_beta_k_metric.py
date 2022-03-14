
from metric import AbstractMetric
from sklearn.metrics import precision_score, recall_score
from util import round_
from random import sample
import logging
from ..pred_true_values_sampler import PredTrueValuesSampler


class FBetaKMetric(AbstractMetric):
    def __init__(self, beta=1, k=10, decimals=None, average='macro', rating_desimals=0):         
        name = f'F{beta}Score@{k}'
        if average != 'macro':
            name += f'({average})'

        super().__init__(name, decimals)

        self._beta = beta
        self._average = average
        self._rating_desimals = rating_desimals
        self._sampler = PredTrueValuesSampler(k, rating_desimals)

    def _calculate(self, pred_values, true_values, opts={}):
        pred_values, true_values = self._sampler.sample(pred_values, true_values)

        precision_k = self._precision(pred_values, true_values)
        recall_k    = self._recall(pred_values, true_values)

        return  (1 + self._beta**2) / (precision_k**-1 + (self._beta**2 / recall_k))

    def _precision(self, pred_values, true_values):
        score = precision_score(pred_values, true_values, average=self._average, zero_division=0)
        return 10**-10 if score == 0 else score 

    def _recall(self, pred_values, true_values):
        score = precision_score(pred_values, true_values, average=self._average, zero_division=0)
        return 10**-10 if score == 0 else score 
