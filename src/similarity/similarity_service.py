from sklearn.neighbors import NearestNeighbors
from abc import ABC, abstractmethod
import numpy as np
from .similarity import adjusted_cosine_sim


class SimilarityService(ABC):
    @abstractmethod
    def similars(self, rm_row, n_neighbors):
        pass

class CommonSimilarityService(SimilarityService):
    def __init__(self, rm, metric, n_neighbors, algorithm = 'brute'):
        self.knn_model = NearestNeighbors(metric = metric, algorithm = algorithm) 
        self.knn_model.fit(rm.data)
        self.rm = rm
        self.n_neighbors = n_neighbors

    def similars(self, row_id, exclude_source_row=True):
        n_neighbors = self.n_neighbors

        if exclude_source_row:
           n_neighbors += 1

        distances, indices = self.knn_model.kneighbors(
            self.rm.row(row_id), 
            n_neighbors=n_neighbors
        )

        similatiry, indices = 1 - distances.flatten(), indices.flatten()

        return (similatiry[1:], indices[1:]) if exclude_source_row else (similatiry, indices)


class AdjustedCosineSimilarityService(SimilarityService):
    def __init__(self, rm, n_neighbors):
        self.distances_matrix = adjusted_cosine_sim(rm)
        self.n_neighbors = n_neighbors

    def similars(self, row_id, exclude_source_row=True):
        similarities = self.distances_matrix[row_id-1].sort_values(ascending=False)
        indices      = self.distances_matrix[row_id-1].sort_values(ascending=False)

        if exclude_source_row:
            similarities = similarities[1:self.n_neighbors+1]
            indices      = indices[1:self.n_neighbors+1]
        else:
            similarities = similarities[:self.n_neighbors] 
            indices      = indices[:self.n_neighbors]

        return similarities.values, indices.index

