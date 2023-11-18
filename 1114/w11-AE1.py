# Created or modified on Nov 2023
# Author: 임일
# Auto Encoder 1

import pandas as pd
import numpy as np

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
mu = ratings_train.rating.mean()    # 전체 평균 

# 사용자, 영화 일련번호 부여하기
user_idx = {}
movie_idx = {}
for i in set(ratings.user_id):
    user_idx[i] = len(user_idx)
for i in set(ratings.movie_id):
    movie_idx[i] = len(movie_idx)

M = len(user_idx)   # Number of users
N = len(movie_idx)  # Number of movies

# 입력 데이터 준비
train_data = np.zeros((M, N))
for rating in np.array(ratings_train):
    train_data[user_idx[rating[0]], movie_idx[rating[1]]] = rating[2] - mu

test_data = np.zeros((M, N))
for rating in np.array(ratings_test):
    test_data[user_idx[rating[0]], movie_idx[rating[1]]] = rating[2] - mu   #test data에서 rating이 된 값만 (mu 빼기)

# DL 모델
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Masking, Activation, Flatten
from tensorflow.keras.optimizers import Adamax, SGD
from tensorflow.keras import regularizers

# Variable 초기화 
H1 = 4096                           # Hidden node 수 1
H2 = 1024                           # Hidden node 수 2
reg = 0.0005                        # Regularization penalty
BATCH_SIZE = 256                    # Batch size

# Defining RMSE measure
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Keras model
input1 = Input(shape=(N))
layer1 = Dense(H1)(input1)
layer1 = Activation('linear')(layer1)
layer1 = Dense(H2, kernel_regularizer=regularizers.L1(reg))(layer1)
layer1 = Activation('linear')(layer1)
layer1 = Dense(H1)(layer1)
layer1 = Activation('linear')(layer1)
layer1 = Dense(N)(layer1)
layer1 = Flatten()(layer1)
model = Model(inputs=input1, outputs=layer1)
model.summary()

model.compile(
  loss=RMSE,
  #optimizer=SGD(lr=0.05, momentum=0.9),
  optimizer=Adamax(lr=0.0003),
  metrics=[RMSE]
)

# Model fitting
result = model.fit(
  x=train_data,
  y=train_data,
  epochs=5,
  batch_size=BATCH_SIZE,
  validation_data=(train_data, test_data)
)


# 정확도(RMSE)를 계산하는 함수 
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# Prediction
predictions = np.zeros((M, N))
for i in range(M):          # 예측값 가져오기
    prediction = model.predict(train_data[np.newaxis, i])
    predictions[i] = prediction.ravel() + mu
y_true = []
y_pred = []
for rating in np.array(ratings_test):
    y_true.append(rating[2])
    y_pred.append(predictions[user_idx[rating[0]], movie_idx[rating[1]]]) 
print("Actuals: \n", (y_true[:20]))
print("Predictions: \n", y_pred[:20])

predictions[predictions > 5] = 5
predictions[predictions < 1] = 1
print(RMSE2(y_true, y_pred))
