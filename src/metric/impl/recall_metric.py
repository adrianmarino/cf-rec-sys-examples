from metric import AbstractMetric
from sklearn.metrics import recall_score
from util import round_


class RecallMetric(AbstractMetric):
    def __init__(self, decimals=None, average='macro', rating_desimals=0):
        name = f'Recall'
        if average != 'macro':
            name += f'({average})'
 
        super().__init__(name, decimals)
        self._average = average
        self._rating_desimals = rating_desimals

    def _calculate(self, pred_values, true_values, opts={}):
        pred_values = [round_(v, self._rating_desimals) for v in pred_values]
        true_values = [round_(v, self._rating_desimals) for v in true_values]

        return recall_score(pred_values, true_values, average=self._average, zero_division=0)
