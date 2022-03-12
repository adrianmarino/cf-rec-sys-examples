from abc import ABC, abstractmethod


class AbstractPredictor(ABC):
    def __init__(self, rm, sim_service, name=None):
        self.sim_service = sim_service
        self.rm = rm
        self._name = name

    @property
    def name(self):
        return self._name if self._name else str(self.__class__.__name__)

    @abstractmethod    
    def predict(self, user_id, item_id, decimals=None):
        pass

    def predict_batch(self, batch, decimals=None):
        return [self.predict(batch[i][0], batch[i][1]) for i in range(len(batch))]