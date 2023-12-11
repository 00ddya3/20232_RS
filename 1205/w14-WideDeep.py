# Created or modified on Nov 2023
# author: 임일
# Deep learning 5

import numpy as np
import pandas as pd

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)            # timestamp 제거
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/RecoSys/Data/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users[['user_id', 'occupation']]
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies[['movie_id', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
movies = movies.set_index('movie_id')

# train test 분리
from sklearn.utils import shuffle
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state=12)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.layers import Dense, Concatenate, Activation, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax
from tensorflow.keras.callbacks import ModelCheckpoint

# Convert occupation(string to integer)
occupation = {}
def convert_occ(x):
    if x in occupation:
        return occupation[x]
    else:
        occupation[x] = len(occupation)
        return occupation[x]

users['occupation'] = users['occupation'].apply(convert_occ)
L = len(occupation)

# Add occupation variable
occ_train = pd.merge(ratings_train, users, on='user_id')['occupation']
occ_test = pd.merge(ratings_test, users, on='user_id')['occupation']

# Adding context variables
genre_train = pd.merge(ratings_train, movies, on='movie_id')
genre_train = genre_train.drop(['user_id', 'movie_id', 'rating'], axis=1)
genre_test = pd.merge(ratings_test, movies, on='movie_id')
genre_test = genre_test.drop(['user_id', 'movie_id', 'rating'], axis=1)

# Variable 초기화 
K = 200                             # Latent factor 수 
reg = 0.0001                        # Regularization penalty
mu = ratings_train.rating.mean()    # 전체 평균 
N = len(set(ratings.user_id)) + 1   # Number of users
M = len(set(ratings.movie_id)) + 1  # Number of movies

# Defining RMSE measure
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# MF layer
user = Input(shape=(1, ))                                               # User input
item = Input(shape=(1, ))                                               # Item input
occ = Input(shape=(1,))                                                 # Occupation input
genre = Input(shape=(19,))                                               # Genre input
P_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(user)     # (N, 1, K)
Q_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(item)     # (M, 1, K)
occ_embedding = Embedding(L, 5, embeddings_regularizer=l2(reg))(occ)
genre_embedding = Embedding(19, 5, embeddings_regularizer=l2(reg))(genre)
user_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(user)
item_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(item)

# MF layer
R1 = layers.dot([P_embedding, Q_embedding], axes=2)
R1 = layers.add([R1, user_bias, item_bias])
R1 = Flatten()(R1)
R1 = Dense(128)(R1)
R1 = Activation('linear')(R1)

# DL layer
user_bias = Flatten()(user_bias)
item_bias = Flatten()(item_bias)
P_embedding = Flatten()(P_embedding)
Q_embedding = Flatten()(Q_embedding)
occ_embedding = Flatten()(occ_embedding)
genre_embedding = Flatten()(genre_embedding)
R2 = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias, occ_embedding, genre_embedding])
R2 = Dense(512)(R2)
R2 = Activation('gelu')(R2)
R2 = Dense(1)(R2)
R2 = Activation('linear')(R2)
R2 = Flatten()(R2)

# MF + DL
R = Concatenate()([R1, R2])
R = Dense(512)(R)
R = Activation('gelu')(R)
R = Dense(64)(R)
R = Activation('linear')(R)
R = Dense(1)(R)

model = Model(inputs=[user, item, occ, genre], outputs=R)
model.compile(
  loss=RMSE,
  #optimizer=SGD(learning_rate=0.0003, momentum=0.8),
  optimizer=Adamax(learning_rate=0.00003),
  metrics=[RMSE],
)
model.summary()

checkpoint_path = 'CheckPoint'
checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             monitor='val_RMSE', 
                             verbose=1)

result = model.fit(
  x=[ratings_train.user_id.values, ratings_train.movie_id.values, occ_train, genre_train],
  y=ratings_train.rating.values - mu,
  callbacks=[checkpoint],
  epochs=100,
  batch_size=128,
  validation_data=(
    [ratings_test.user_id.values, ratings_test.movie_id.values, occ_test, genre_test],
    ratings_test.rating.values - mu
  )
)

# Plot RMSE
import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'], label="Train RMSE")
plt.plot(result.history['val_RMSE'], label="Test RMSE")
plt.legend()
plt.show()

model.load_weights(checkpoint_path)

# 정확도(RMSE)를 계산하는 함수 
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

user_ids = ratings_test.user_id.values
movie_ids = ratings_test.movie_id.values
y_pred = model.predict([user_ids, movie_ids, occ_test, genre_test]) + mu
y_pred = np.ravel(y_pred, order='C')
y_true = np.array(ratings_test.rating)

RMSE2(y_true, y_pred)

