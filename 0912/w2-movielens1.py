# Created or modifed on Sep 2023
# author: 임일
# MovieLens best-seller

# Load the u.user file into a dataframe
import pandas as pd
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/RecoSys/Data/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users.set_index('user_id')
users.head()

# Load the u.item file into a dataframe
import pandas as pd
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies.set_index('movie_id')
movies.head()

# Load the u.data file into a dataframe
import pandas as pd
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=r_cols, encoding='latin-1') 
ratings = ratings.set_index('user_id')
ratings.head()

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

def recom_movie2(n_items=5):
    return movies.loc[movie_mean.sort_values(ascending=False)[:n_items].index]['title']

movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
recom_movie1(5)

rmse = []
for user in set(ratings.index):
    y_true = ratings.loc[user][['movie_id', 'rating']]
    y_pred = movie_mean[ratings.loc[user]['movie_id']]
    accuracy = RMSE(y_true['rating'], y_pred)
    rmse.append(accuracy)
print(np.mean(rmse))

















