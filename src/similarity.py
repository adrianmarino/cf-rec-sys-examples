from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from kneighbors import kneighbors

from scipy.stats import spearmanr

def spearman(x, y):
    rho, pval = spearmanr(x, y, axis=0)
    return rho * (-1)

def spearman_sim(rm):
    return pd.DataFrame(1 - pairwise_distances(rm.data, metric=spearman))

def cosine_sim(rm):
    return pd.DataFrame(1 - pairwise_distances(rm.data, metric="cosine"))


def pearson_sim(rm):
    return pd.DataFrame(1 - pairwise_distances(rm.data, metric="correlation"))


def adjusted_cosine_sim(rm):
    M = rm.data
    sim_matrix = np.zeros((M.shape[1], M.shape[1]))
    M_u = M.mean(axis=1) #means
          
    for i in range(M.shape[1]):
        for j in range(M.shape[1]):
            if i == j:
                
                sim_matrix[i][j] = 1
            else:                
                if i<j:
                    
                    sum_num = sum_den1 = sum_den2 = 0
                    for k,row in M.loc[:,[i,j]].iterrows(): 

                        if ((M.loc[k,i] != 0) & (M.loc[k,j] != 0)):
                            num = (M[i][k]-M_u[k])*(M[j][k]-M_u[k])
                            den1= (M[i][k]-M_u[k])**2
                            den2= (M[j][k]-M_u[k])**2
                            
                            sum_num = sum_num + num
                            sum_den1 = sum_den1 + den1
                            sum_den2 = sum_den2 + den2
                        
                        else:
                            continue                          
                                       
                    den=(sum_den1**0.5)*(sum_den2**0.5)
                    if den!=0:
                        sim_matrix[i][j] = sum_num/den
                    else:
                        sim_matrix[i][j] = 0


                else:
                    sim_matrix[i][j] = sim_matrix[j][i]           
            
    return pd.DataFrame(sim_matrix)


def k_similar_rows(rm, row_id, metric, n_neighbors, exclude_source_row=True):
    """
    Return similarities, indexes
    """
    if exclude_source_row:
        n_neighbors += 1

    distances, indices = kneighbors(rm, rm.row(row_id), metric, n_neighbors)
    return (1 - distances[1:], indices[1:]) if exclude_source_row else (1 - distances, indices)


def k_similar_rows_using_adjusted_cosine(rm, row_id, n_neighbors, exclude_source_row=True):
    sim_matrix = adjusted_cosine_sim(rm)

    similarities = sim_matrix[row_id-1].sort_values(ascending=False)
    indices      = sim_matrix[row_id-1].sort_values(ascending=False)

    if exclude_source_row:
        similarities = similarities[1:n_neighbors+1]
        indices      = indices[1:n_neighbors+1]
    else:
        similarities = similarities[:n_neighbors] 
        indices      = indices[:n_neighbors]

    return similarities.values, indices.index