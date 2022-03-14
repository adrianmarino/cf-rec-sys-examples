from random import sample
import logging
from util import round_


class PredTrueValuesSampler:
    def __init__(self, k, rating_desimals): 
        self._k = k
        self._rating_desimals = rating_desimals

    def sample(self, pred_values, true_values):
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
