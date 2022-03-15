
from metric import AbstractMetric
from sklearn.metrics import fbeta_score
from ..user_pred_true_sampler import UserPredTrueSampler
from util import round_


class FBetaMetric(AbstractMetric):
    def __init__(self, beta=1, decimals=4, average='macro', rating_decimals=0):         
        name = f'F{beta}Score'
        if average != 'macro':
            name += f'({average})'

        super().__init__(name, decimals)

        self._beta            = beta
        self._average         = average
        self._rating_decimals = rating_decimals

    def _calculate(self, pred_values, true_values, ctx):
        true_values = [round_(v, self._rating_decimals) for v in true_values]
        pred_values = [round_(v, self._rating_decimals) for v in pred_values]

        return fbeta_score(true_values, pred_values, beta=self._beta, average=self._average)
