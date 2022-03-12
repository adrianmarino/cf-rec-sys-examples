from bunch import Bunch


class EvaluatorContext:
    def __init__(self, rm):
        self.true_values = []
        self.predictors = Bunch()
        self.rm = rm

    def add_pred(self, predictor, pred_values):
        if predictor.name not in self.predictors:
            self.predictors[predictor.name] = Bunch(pred_values=[], predictor=predictor)

        self.predictors[predictor.name].pred_values.extend(pred_values)

    def add_true_values(self, values):
        self.true_values.extend(values)
