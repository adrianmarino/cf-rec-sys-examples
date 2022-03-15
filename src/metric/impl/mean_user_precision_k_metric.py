from metric import AbstractMetric
from sklearn.metrics import precision_score
from ..user_pred_true_sampler import UserPredTrueSampler
import statistics


class MeanUserPrecisionKMetric(AbstractMetric):
    def __init__(self, k=10, decimals=4, average='macro', rating_decimals=0):
        name = f'MeanUserPrecision@{k}'
        if average != 'macro':
            name += f'({average})'

        super().__init__(name, decimals)

        self._average         = average
        self._rating_decimals = rating_decimals
        self._sampler         = UserPredTrueSampler(k, rating_decimals)


    def _calculate(self, pred_values, true_values, opts):
        scores = []
        for user_pred_values, user_true_values in self._sampler.sample(opts.ctx, opts.predictor_name):
            scores.append(precision_score(user_true_values, user_pred_values, average=self._average, zero_division=0))
        return statistics.mean(scores)

