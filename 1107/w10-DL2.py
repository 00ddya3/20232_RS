# Created or modified on Nov 2023
# author: 임일
# Deep learning 2 (More layers)

import numpy as np
import pandas as pd

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)            # timestamp 제거

# train test 분리
from sklearn.utils import shuffle
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state=12)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.layers import Dense, Concatenate, Activation, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax
from tensorflow.keras.callbacks import ModelCheckpoint

# Variable 초기화 
K = 350                             # Latent factor 수 
reg = 0.005                         # Regularization penalty
mu = ratings_train.rating.mean()    # 전체 평균 
N = len(set(ratings.user_id)) + 1   # Number of users
M = len(set(ratings.movie_id)) + 1  # Number of movies

# Defining RMSE measure
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Keras model
user = Input(shape=(1, ))                                               # User input
item = Input(shape=(1, ))                                               # Item input
P_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(user)     # (N, 1, K)
Q_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(item)     # (M, 1, K)
user_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(user)       # User bias term (N, 1, )
item_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(item)       # Item bias term (M, 1, )
P_embedding = Flatten()(P_embedding)                                    # (N, K)
Q_embedding = Flatten()(Q_embedding)                                    # (N, K)
user_bias = Flatten()(user_bias)                                        # (N, K)
item_bias = Flatten()(item_bias)                                        # (N, K)

R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias])     # (N, 2K + 2)

# Neural network
R = Dense(2048)(R)
R = Activation('gelu')(R)
R = Dense(1024)(R)
R = Activation('LeakyReLU')(R)
R = Dense(512)(R)
R = Activation('linear')(R)
R = Dense(1)(R)

model = Model(inputs=[user, item], outputs=R)
model.compile(
  loss=RMSE,
  #optimizer=SGD(lr=0.003, momentum=0.9),
  optimizer=Adamax(lr=0.0002),
  metrics=[RMSE],
)
model.summary()

checkpoint_path = 'CheckPoint'
checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             monitor='val_RMSE', 
                             verbose=1)

# Model fitting
result = model.fit(
  x=[ratings_train.user_id.values, ratings_train.movie_id.values],
  y=ratings_train.rating.values - mu,
  callbacks=[checkpoint],
  epochs=15,
  batch_size=128,
  validation_data=(
    [ratings_test.user_id.values, ratings_test.movie_id.values],
    ratings_test.rating.values - mu
  )
)

model.load_weights(checkpoint_path)

# Plot RMSE
import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'], label="Train RMSE")
plt.plot(result.history['val_RMSE'], label="Test RMSE")
plt.legend()
plt.show()

# 정확도(RMSE)를 계산하는 함수 
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

user_ids = ratings_test.user_id.values
movie_ids = ratings_test.movie_id.values
y_pred = model.predict([user_ids, movie_ids]) + mu
y_pred = np.ravel(y_pred, order='C')
y_true = np.array(ratings_test.rating)

RMSE2(y_true, y_pred)



