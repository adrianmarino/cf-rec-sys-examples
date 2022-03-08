from similarity.service.similarity_service import SimilarityService
from similarity.measure import adjusted_cosine_sim

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
