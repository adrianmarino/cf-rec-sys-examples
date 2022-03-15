from metric import AbstractMetric
from ..user_pred_true_sampler import UserPredTrueSampler
from ..binarizer import identity
import statistics


class MeanAVGPrecisionKMetric(AbstractMetric):
    def __init__(self, k=10, decimals=4, binarizer=identity(), average='macro', rating_decimals=0):
        super().__init__(f'MeanAVGPrecision@{k}{binarizer.desc}', decimals)
        self._sampler = UserPredTrueSampler(k, rating_decimals) 
        self._binarize_fn = binarizer.closure()
        self._k = k
        self._average = average


    def _calculate(self, pred_values, true_values, opts):
        scores = []
        for user_pred_values, user_true_values in self._sampler.sample(opts.ctx, opts.predictor_name):
            true_positives_count = sum([self._binarize_fn(v) for v in user_true_values])

            if true_positives_count == 0:
                scores.append(0)
            else:
                tp_sum = sum([self._binarize_fn(v)/(i+1) for i, v in enumerate(user_true_values)])
                scores.append(tp_sum / min(self._k, true_positives_count))

        return statistics.mean(scores)
