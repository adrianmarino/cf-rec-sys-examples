from sklearn.neighbors import NearestNeighbors


def kneighbors(rm, rm_row, metric, n_neighbors):
    knn_model = NearestNeighbors(metric = metric, algorithm = 'brute') 
    knn_model.fit(rm.data)

    distances, indices = knn_model.kneighbors(rm_row, n_neighbors =n_neighbors)
    return 1 - distances.flatten(), indices.flatten()