# Created or modified on Nov 2023
# author: 임일
# Tensorflow/Keras 예

import tensorflow as tf
import keras

(train_data, train_targets), (test_data, test_targets) = keras.datasets.boston_housing.load_data()
print(train_data)

print(train_data.shape)
print(test_data.shape)

train_data[:5,]
train_targets[:5]

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
train_data[:5]

test_data -= mean
test_data /= std

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation = 'ReLU', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation = 'ReLU'))
model.add(layers.Dense(1))      # y가 연속값이므로 node가 하나임
model.summary()

adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss='mse', metrics=['mean_squared_error'])
# model.fit(train_data, train_targets, epochs=60, batch_size=64)
result = model.fit(train_data, train_targets, epochs=30, batch_size=64, validation_data=(test_data, test_targets))

min(result.history['val_mean_squared_error'])
y_pred = model.predict(test_data)

# 성능평가
from sklearn.metrics import r2_score
import numpy as np

y_pred = np.ravel(y_pred)
r2_score(test_targets, y_pred)
print(test_targets, y_pred)

import matplotlib.pyplot as plt
plt.plot(result.history['mean_squared_error'], label="Train MSE")
plt.plot(result.history['val_mean_squared_error'], label="Test MSE")
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
















