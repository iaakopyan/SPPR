import numpy as np
import pandas as pd

ratings_df = pd.read_csv("ratings.csv")
print('Unique users count: {}'.format(len(ratings_df['userId'].unique())))
print('Unique books count: {}'.format(len(ratings_df['bookId'].unique())))
print('DataFrame shape: {}'.format(ratings_df.shape))

ratings_df.head()

n = 95000
ratings_df_sample = ratings_df[:n]
n_users = len(ratings_df_sample["userId"].unique())
n_movies = len(ratings_df_sample["bookId"].unique())

print(f'юзеры: {n_users}, фильмы: {n_movies}')

movies = ratings_df["bookId"].unique()


def scale_movie_id(movie_id):
    scaled = np.where(movie_id == movie_id)[0][0] + 1
    return scaled


ratings_df['bookId'] = ratings_df['bookId'].apply(scale_movie_id)
print(ratings_df.head(3))
