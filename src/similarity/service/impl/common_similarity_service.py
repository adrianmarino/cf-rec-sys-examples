from sklearn.neighbors import NearestNeighbors
from similarity.service.similarity_service import SimilarityService

class CommonSimilarityService(SimilarityService):
    def __init__(self, rm, metric, n_neighbors, algorithm = 'brute', metric_params=None):
        self.knn_model = NearestNeighbors(metric = metric, algorithm = algorithm, metric_params=metric_params)
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

