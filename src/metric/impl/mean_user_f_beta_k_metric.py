
from metric import AbstractMetric
from sklearn.metrics import fbeta_score
from ..user_pred_true_sampler import UserPredTrueSampler
import statistics


class MeanUserFBetaKMetric(AbstractMetric):
    def __init__(self, beta=1, k=10, decimals=4, average='macro', rating_decimals=0):         
        name = f'MeanUserF{beta}Score@{k}'
        if average != 'macro':
            name += f'({average})'

        super().__init__(name, decimals)

        self._beta            = beta
        self._average         = average
        self._rating_decimals = rating_decimals
        self._sampler         = UserPredTrueSampler(k, rating_decimals)


    def _calculate(self, pred_values, true_values, opts):
        scores = []
        for user_pred_values, user_true_values in self._sampler.sample(opts.ctx, opts.predictor_name):
            scores.append(fbeta_score(user_true_values, user_pred_values, average=self._average, beta=self._beta))
        return statistics.mean(scores)
