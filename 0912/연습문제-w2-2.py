# Created or modifed on Sep 2023
# Author: 임일
# Each Movie

import pandas as pd

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/vote.csv', encoding='utf-8')
ratings = ratings[['Person', 'Movie_ID', 'Score']]
ratings.columns = ['user_id', 'movie_id', 'rating']
ratings = ratings.replace({0.:1, 0.2:1, 0.4:2, 0.6:3, 0.8:4, 1.:5})
ratings = ratings.dropna()
ratings = ratings.drop_duplicates(subset=['user_id', 'movie_id'])
ratings = ratings.set_index('user_id')

from sklearn.utils import shuffle
ratings = shuffle(ratings)
ratings = ratings.iloc[:250000]

ratings_by_user = ratings.groupby(by='user_id')['rating'].count()
ratings_by_user = ratings_by_user[ratings_by_user > 10]
ratings = ratings.loc[ratings_by_user.index]
ratings.to_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')

movies = pd.read_csv('C:/RecoSys/Data/movie.csv', encoding='utf-8')
movies = movies[['ID', 'Name']]
movies.columns = ['movie_id', 'title']
movies = movies.set_index('movie_id')

# Best-seller recommender
import numpy as np

def RMSE(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def recom_movie1(n_items=5):
    movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
    recom_movies = movies.loc[movie_sort.index]
    recommendations = recom_movies['title']
    return recommendations

movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
recom_movie1(10)

rmse = []
for user in set(ratings.index):
    y_true = ratings.loc[user][['movie_id', 'rating']]
    y_pred = movie_mean[ratings.loc[user]['movie_id']]
    accuracy = RMSE(y_true['rating'], y_pred)
    rmse.append(accuracy)
print(np.mean(rmse))
