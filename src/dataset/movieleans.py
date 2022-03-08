import pandas as pd
from util import most_common, path_exists, Sequence
import os

DATASET_PATH = './ml-latest-small'

class MovielensDataset:
    
    def __init__(self, n_users=200, n_movies=200):
        if not path_exists(DATASET_PATH):
           os.system('wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip')
           os.system('unzip ml-latest-small.zip')
        self.n_users = n_users
        self.n_movies = n_movies

    def __preprocessing(self, df, n_users, n_movies):
        pd.options.mode.chained_assignment = None 
        most_common_users  = most_common(df, 'userId', n_users)
        most_common_movies = most_common(df, 'movieId', n_movies)
        df = df[df['userId'].isin(most_common_users) & df['movieId'].isin(most_common_movies)]

        df['user_seq']  = df.userId.apply(Sequence(1).apply)
        df['movie_seq'] = df.movieId.apply(Sequence(1).apply)

        return df.drop(columns=['userId', 'timestamp'])

    def ratings(self):
        df = pd.read_csv(f'{DATASET_PATH}/ratings.csv')
        return self.__preprocessing(df, self.n_users, self.n_movies)