from similarity import k_similar_rows, \
                       k_similar_rows_using_adjusted_cosine, \
                       spearman
import numpy as np
from scipy.spatial.distance import correlation, cosine
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt




def item_based_rating_predict(rm, user_id, item_id, metric, n_neighbors):
    t_rm = rm.T()
    row_sims, row_indices = k_similar_rows(t_rm, item_id, metric, n_neighbors)
    numerator = np.sum([t_rm.cell(row_id, user_id) * sim for row_id, sim in zip(row_indices+1, row_sims)])
    return round(numerator / np.sum(row_sims), 2)


def item_based_adjcos_rating_predict(rm, user_id, item_id, n_neighbors):
    row_sims, row_indices = k_similar_rows_using_adjusted_cosine(rm, item_id, n_neighbors)
    numerator = np.sum([rm.cell(user_id, row_id) * sim for row_id, sim in zip(row_indices+1, row_sims)])
    return np.clip(round(numerator / np.sum(row_sims), 2), 1, 10)


def user_based_rating_predict(rm, user_id, item_id, metric, n_neighbors):
    row_sims, row_indices = k_similar_rows(rm, user_id, metric, n_neighbors)
    wtd_sum = np.sum([
        rm.row_deviation(curr_row_id, item_id) * sim \
            for curr_row_id, sim in zip(row_indices+1, row_sims)
    ])
    return round(rm.mean_row(user_id) + (wtd_sum / np.sum(row_sims)), 2)


def create_predictors(rm, n_neighbors):
    return {
        'user_minkowski': lambda user_id, item_id: user_based_rating_predict(rm, user_id, item_id, 'minkowski', n_neighbors),
        'user_euclidean': lambda user_id, item_id: user_based_rating_predict(rm, user_id, item_id, 'euclidean', n_neighbors),
        'user_cosine':    lambda user_id, item_id: user_based_rating_predict(rm, user_id, item_id, cosine, n_neighbors),
        'user_pearson':   lambda user_id, item_id: user_based_rating_predict(rm, user_id, item_id, correlation, n_neighbors),
        'user_spearman':   lambda user_id, item_id: user_based_rating_predict(rm, user_id, item_id, spearman, n_neighbors),
        'user_manhattan': lambda user_id, item_id: user_based_rating_predict(rm, user_id, item_id, 'manhattan', n_neighbors),
        'item_minkowski': lambda user_id, item_id: item_based_rating_predict(rm, user_id, item_id, 'minkowski', n_neighbors),
        'item_euclidean': lambda user_id, item_id: item_based_rating_predict(rm, user_id, item_id, 'euclidean', n_neighbors),
        'item_manhattan': lambda user_id, item_id: item_based_rating_predict(rm, user_id, item_id, 'manhattan', n_neighbors),
        'item_cosine':    lambda user_id, item_id: item_based_rating_predict(rm, user_id, item_id, cosine, n_neighbors),
        'item_adj_cosine':lambda user_id, item_id: item_based_adjcos_rating_predict(rm, user_id, item_id, n_neighbors),
        'item_pearson':   lambda user_id, item_id: item_based_rating_predict(rm, user_id, item_id, correlation, n_neighbors),
        'item_spearman':   lambda user_id, item_id: item_based_rating_predict(rm, user_id, item_id, spearman, n_neighbors)
    } 


def predict(predictors, user_id, item_id):
    return {predictor_name: predictor(user_id, item_id) for predictor_name, predictor in predictors.items()}

def metrics_fn(prediction, rm, decimals=3):
    MSE = mean_squared_error(prediction, rm.data)
    return { 
        'rmse': round(sqrt(MSE), decimals), 
        'mse': round(MSE, decimals)
    }

def evaluate(predictors, rm, metrics_fn=metrics_fn):
    results = {}
    for name, predictor in predictors.items():
        prediction = pd.DataFrame(np.zeros((rm.n_rows, rm.n_columns)))

        def predict_fn(_, user_id, item_id): prediction[user_id-1][item_id-1] = predictor(user_id, item_id)
        rm.for_each(predict_fn)

        results[name] = metrics_fn(prediction, rm)

    return results