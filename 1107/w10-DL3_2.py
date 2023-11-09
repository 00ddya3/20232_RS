# Created or modified on Nov 2023
# author: 임일
# Deep learning 3 (other variables)

import pandas as pd
import numpy as np

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)            # timestamp 제거
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/RecoSys/Data/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users[['user_id', 'sex', 'occupation']]


# train test 분리
from sklearn.utils import shuffle
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state=12)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

#Convert sex(string to integer)
users['sex'] = users['sex'].apply(lambda x : 0 if x=='M' else 1)
L_gen = 2

# Convert occupation(string to integer)
occupation = {}
def convert_occ(x):
    if x in occupation:
        return occupation[x]
    else:
        occupation[x] = len(occupation)
        return occupation[x]

users['occupation'] = users['occupation'].apply(convert_occ)
L_occ = len(occupation)

train_so = pd.merge(ratings_train, users, on='user_id')[['sex', 'occupation']]
test_so = pd.merge(ratings_test, users, on='user_id')[['sex', 'occupation']]
##train_gen

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.layers import Dense, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.callbacks import ModelCheckpoint

# Variable 초기화 
K = 200                             # Latent factor 수 
reg = 0.002                         # Regularization penalty
mu = ratings_train.rating.mean()    # 전체 평균 
N = len(set(ratings.user_id)) + 1   # Number of users
M = len(set(ratings.movie_id)) + 1  # Number of movies

# Defining RMSE measure
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Keras model
user = Input(shape=(1, ))
item = Input(shape=(1, ))
P_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(user)     # (N, 1, K)
Q_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(item)     # (M, 1, K)
user_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(user)       # User bias term (N, 1, )
item_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(item)       # Item bias term (M, 1, )

# Concatenate layers
P_embedding = Flatten()(P_embedding)
Q_embedding = Flatten()(Q_embedding)
user_bias = Flatten()(user_bias)
item_bias = Flatten()(item_bias)

gen = Input(shape=(1, ))
gen_embedding = Embedding(L_gen, 2, embeddings_regularizer=l2())(gen)
gen_layer = Flatten()(gen_embedding)
occ = Input(shape=(1, ))
occ_embedding = Embedding(L_occ, 3, embeddings_regularizer=l2())(occ)
occ_layer = Flatten()(occ_embedding)
R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias, gen_layer, occ_layer])  #350+350+1+1+2+3

# Neural network
R = Dense(4096)(R)
R = Activation('gelu')(R)
R = Dense(2048)(R)
R = Activation('LeakyReLU')(R)
R = Dense(512)(R)
R = Activation('linear')(R)
R = Dense(1)(R)

model = Model(inputs=[user, item, gen, occ], outputs=R)
model.compile(
  loss=RMSE,
  #optimizer=SGD(lr=0.003, momentum=0.9),
  optimizer=Adamax(lr=0.0002),
  metrics=[RMSE]
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
  x=[ratings_train.user_id.values, ratings_train.movie_id.values, train_so.sex.values, train_so.occupation.values],
  y=ratings_train.rating.values - mu,
  callbacks=[checkpoint],
  epochs=15,
  batch_size=128,
  validation_data=(
    [ratings_test.user_id.values, ratings_test.movie_id.values, test_so.sex.values, test_so.occupation.values],
    ratings_test.rating.values - mu
  )
)

model.load_weights(checkpoint_path)

# Plot RMSE
import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'], label="Train RMSE")
plt.plot(result.history['val_RMSE'], label="Test RMSE")
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()

# Prediction
user_ids = ratings_test.user_id.values[0:6]
movie_ids = ratings_test.movie_id.values[0:6]
user_so = test_so[0:6]
predictions = model.predict([user_ids, movie_ids, user_so]) + mu
print("Actuals: \n", ratings_test[0:6])
print()
print("Predictions: \n", predictions)

# 정확도(RMSE)를 계산하는 함수 
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

user_ids = ratings_test.user_id.values
movie_ids = ratings_test.movie_id.values
y_pred = model.predict([user_ids, movie_ids, test_so]) + mu
y_pred = np.ravel(y_pred, order='C')
y_true = np.array(ratings_test.rating)

RMSE2(y_true, y_pred)