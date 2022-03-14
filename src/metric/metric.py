import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from util import round_


class AbstractMetric(ABC):
    def __init__(self, name, decimals=None):
        self.name = name
        self._decimals = decimals

    def perform(self, pred_values, true_values, opts={}):
        metric = self._calculate(pred_values, true_values, opts)
        return round_(metric, self._decimals)

    def _calculate(self, pred_values, true_values, opts={}):
        pass

