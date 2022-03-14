from metric import AbstractMetric
from util import round_
from random import sample
import logging
from ..pred_true_values_sampler import PredTrueValuesSampler


class AVGPrecisionKMetric(AbstractMetric):
    def __init__(self, k=10, decimals=None, rating_desimals=0):
        super().__init__(f'AVGPrecision@{k}', decimals)
        self._k = k
        self._sampler = PredTrueValuesSampler(k, rating_desimals)

    def _calculate(self, pred_values, true_values, opts={}):
        pred_values, true_values = self._sampler.sample(pred_values, true_values)
        return apk(pred_values, true_values, self._k)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)
