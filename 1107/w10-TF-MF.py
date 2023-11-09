# Created or modified on Nov 2023
# author: 임일
# MF using Tensorflow

import pandas as pd

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)

N = len(set(ratings.user_id)) + 1      # Number of users
M = len(set(ratings.movie_id)) + 1     # Number of movies
TRAIN_SIZE = 0.75

# train test 분리
from sklearn.utils import shuffle
ratings = shuffle(ratings, random_state=12)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dot, Add, Embedding, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, Adamax

# Variable 초기화 
K = 350                             # Latent factor 수 
mu = ratings_train.rating.mean()    # 전체 평균 
reg = 0.0001                        # Regularization penalty

def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Keras model
user = Input(shape=(1,))                                                # User input
item = Input(shape=(1,))                                                # Item input
P_embedding = Embedding(N, K, embeddings_regularizer=l2(reg))(user)     # (N, 1, K)
Q_embedding = Embedding(M, K, embeddings_regularizer=l2(reg))(item)     # (M, 1, K)

user_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(user)       # User bias term (N, 1, 1)
item_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(item)       # Item bias term (M, 1, 1)

#R = Dot(axes=2)([P_embedding, Q_embedding])                            # (N, 1, 1)
R = layers.dot([P_embedding, Q_embedding], axes=2)
#R = Add()([R, user_bias, item_bias])
R = layers.add([R, user_bias, item_bias])
R = Flatten()(R)                                                        # (N, 1)

# Model setting
model = Model(inputs=[user, item], outputs=R)
model.compile(
  loss=RMSE,
  #optimizer=SGD(lr=0.003, momentum=0.9),
  optimizer=Adamax(lr=0.005),
  metrics=[RMSE],
)
model.summary()

# Model fitting
result = model.fit(
  x=[ratings_train.user_id.values, ratings_train.movie_id.values],
  y=ratings_train.rating.values - mu,
  epochs=85,
  batch_size=128,
  validation_data=(
    [ratings_test.user_id.values, ratings_test.movie_id.values],
    ratings_test.rating.values - mu
  )
)

# Plot RMSE
import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'], label="Train RMSE")
plt.plot(result.history['val_RMSE'], label="Test RMSE")
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()

# Prediction
submodel = Model([user, item], R)
user_ids = ratings_test.user_id.values[0:5]
movie_ids = ratings_test.movie_id.values[0:5]
predictions = submodel.predict([user_ids, movie_ids]) + mu
print("Predictions:", predictions)
ratings_test[0:5]
