from abc import ABC, abstractmethod


class SimilarityService(ABC):
    @abstractmethod
    def similars(self, rm_row, n_neighbors):
        pass
