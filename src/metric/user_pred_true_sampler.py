from random import sample
import logging
from util import round_
import numpy as np


class UserPredTrueSampler:
    def __init__(self, k, decimals=None): 
        self._k = k
        self._decimals = decimals

    def sample(self, ctx, predictor_name):
        for user_id in ctx.user_ids:
            true_values, pred_values = ctx.true_pred_by_user(user_id, predictor_name)
            k_true_values, k_pred_values = self._k_sample(true_values, pred_values)

            k_ordered_indexes = np.argsort(k_true_values)[::-1]
            yield k_true_values[k_ordered_indexes], k_pred_values[k_ordered_indexes]


    def _k_sample(self, true_values, pred_values):
        values_size  = len(true_values)
        k = self._k

        if k > values_size:
            logging.debug(f'Use K={values_size} because defined k={self._k} is greater than values size ({values_size}).')
            k = values_size

        indexes = np.array(sample(list(range(values_size)), k))

        k_pred_values = np.array([round_(pred_values[i], self._decimals) for i in indexes])
        k_true_values = np.array([round_(true_values[i], self._decimals) for i in indexes])

        return k_true_values, k_pred_values
