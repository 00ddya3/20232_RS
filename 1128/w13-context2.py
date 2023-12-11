# Created or modified on Nov 2023
# Author: 임일
# Context-aware hybrid (MF + CF bagging + Context variables)


import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, Multiply, Concatenate, Embedding
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, Activation, Flatten
from tensorflow.keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.callbacks import ModelCheckpoint

# csv 파일에서 불러오기
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies[['movie_id', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
movies = movies.set_index('movie_id')

# Load the u.user file into a dataframe
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/RecoSys/Data/u.user', sep='|', names=u_cols, encoding='latin-1')

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

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)

N = ratings.user_id.max() + 1       # Number of users
M = ratings.movie_id.max() + 1      # Number of movies

# train test 분리
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state=12)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

context_train1 = pd.merge(ratings_train, movies, on='movie_id')      # Adding context variables
context_train1 = context_train1.drop(['user_id', 'movie_id', 'rating'], axis=1)
context_test1 = pd.merge(ratings_test, movies, on='movie_id')
context_test1 = context_test1.drop(['user_id', 'movie_id', 'rating'], axis=1)

context_train2 = pd.merge(ratings_train, users, on='user_id')      # Adding context variables
context_train2 = context_train2[['occupation']]
context_test2 = pd.merge(ratings_test, users, on='user_id')
context_test2 = context_test2[['occupation']]

ratings = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)

# Predictions using MF ###########################################################################
class NEW_MF():
    # Initializing the object
    def __init__(self, ratings, K, alpha, beta, iterations, tolerance=0.005, verbose=True):
        self.R = np.array(ratings)
        # user_id, movie_id를 R의 index와 매칭하기 위한 dictionary 생성
        item_id_index = []
        index_item_id = []
        for i, one_id in enumerate(ratings):
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        self.item_id_index = dict(item_id_index)
        self.index_item_id = dict(index_item_id)        
        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)
        # 다른 변수 초기화
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.tolerance = tolerance
        self.verbose = verbose

    # 테스트 셋을 선정하는 메소드 
    def set_test(self, ratings_test):                           # Setting test set
        test_set = []
        for i in range(len(ratings_test)):                      # Selected ratings
            x = self.user_id_index[ratings_test.iloc[i,0]]      # Getting R indice for the given user_id and movie_id
            y = self.item_id_index[ratings_test.iloc[i,1]]
            z = ratings_test.iloc[i,2]
            test_set.append([x, y, z])
            self.R[x, y] = 0                    # Setting test set ratings to 0
        self.test_set = test_set
        return test_set                         # Return test set

    def test(self):                             # Training 하면서 test set의 정확도를 계산하는 메소드 
        # Initializing user-feature and movie-feature matrix
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])

        # List of training samples
        rows, columns = self.R.nonzero()
        self.samples = [(i,j, self.R[i,j]) for i, j in zip(rows, columns)]

        # Stochastic gradient descent for given number of iterations
        best_RMSE = 10000
        best_iteration = 0
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.6f ; Test RMSE = %.6f" % (i+1, rmse1, rmse2))
            if best_RMSE > rmse2:                      # New best record
                best_RMSE = rmse2
                best_iteration = i
            elif (rmse2 - best_RMSE) > self.tolerance: # RMSE is increasing over tolerance
                break
        print(best_iteration, best_RMSE)
        return training_process

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])

    # Computing mean squared error
    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []
        for x, y in zip(xs, ys):
            prediction = self.get_prediction(x, y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x, y] - prediction)
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    # Test RMSE 계산하는 method 
    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.get_prediction(one_set[0], one_set[1])
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))

    # Ratings for user i and moive j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Ratings for user_id and moive_id
    def get_one_prediction(self, user_id, movie_id):
        return self.get_prediction(self.user_id_index[user_id], self.item_id_index[movie_id])

# Creating MF Object & train
ratings_temp = ratings.copy()
mf = NEW_MF(ratings_temp, K=200, alpha=0.0014, beta=0.075, iterations=320, tolerance=0.0001, verbose=True)
test_set = mf.set_test(ratings_test)
result = mf.test()


# Predictions using CF ###########################################################################
from sklearn.metrics.pairwise import cosine_similarity
rating_matrix = ratings_train.pivot_table(values='rating', index='user_id', columns='movie_id')

# 모든 가능한 사용자 pair의 Cosine similarities 계산
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# 모든 user의 rating 평균 계산 
rating_mean = rating_matrix.mean(axis=1)

def ubcf_bias_knn(user_id, movie_id, neighbor_size=0):
    # 현 user의 평균 가져오기
    user_mean = rating_mean[user_id]
    if movie_id in rating_matrix:
        # 현 user와 다른 사용자 간의 유사도 가져오기
        sim_scores = user_similarity[user_id]
        # 현 movie의 rating 가져오기
        movie_ratings = rating_matrix[movie_id]
        # 모든 사용자의 rating 평균 가져오기
        others_mean = rating_mean
        # 현 movie에 대한 rating이 없는 user 삭제
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        others_mean = others_mean.drop(none_rating_idx)
        if neighbor_size == 0:               # Neighbor size가 지정되지 않은 경우
            # 편차로 예측치 계산
            movie_ratings = movie_ratings - others_mean
            prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            # 예측값에 현 사용자의 평균 더하기
            prediction = prediction + user_mean
        else:                                # Neighbor size가 지정된 경우
            # 지정된 neighbor size 값과 해당 영화를 평가한 총사용자 수 중 작은 것으로 결정
            neighbor_size = min(neighbor_size, len(sim_scores))
            # array로 바꾸기 (argsort를 사용하기 위함)
            sim_scores = np.array(sim_scores)
            movie_ratings = np.array(movie_ratings)
            others_mean = np.array(others_mean)
            # 유사도를 순서대로 정렬
            user_idx = np.argsort(sim_scores)
            # 유사도와 rating을 neighbor size만큼 받기
            sim_scores = sim_scores[user_idx][-neighbor_size:]
            movie_ratings = movie_ratings[user_idx][-neighbor_size:]
            # 사용자의 mean을 neighbor size만큼 받기
            others_mean = others_mean[user_idx][-neighbor_size:]
            # 편차로 예측치 계산
            movie_ratings = movie_ratings - others_mean
            if sim_scores.sum() > 0.00001:
                prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum() + user_mean
            else:
                prediction = user_mean
    else:
        prediction = user_mean
    return prediction

# Context-aware recommendation ###########################################################################
def recommender0(recomm_list, mf):
    id_pairs = zip(recomm_list[:, 0], recomm_list[:, 1])
    recommendations = np.array([mf.get_one_prediction(user, movie) for (user, movie) in id_pairs])
#    recommendations = []
#    for i in range(len(recomm_list)):
#        recommendations.append(mf.get_one_prediction(recomm_list[i,0], recomm_list[i,1]))
    return recommendations

def recommender1(recomm_list, neighbor_size=0):
    id_pairs = zip(recomm_list[:, 0], recomm_list[:, 1])
    recommendations = np.array([ubcf_bias_knn(user, movie, neighbor_size) for (user, movie) in id_pairs])
    return recommendations

# RMSE 계산을 위한 함수
def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

recomm_list = np.array(ratings_train.iloc[:, [0, 1]])       # Data for training context-DL model
train0 = recommender0(recomm_list, mf)
train1 = recommender1(recomm_list, 37)

recomm_list = np.array(ratings_test.iloc[:, [0, 1]])        # Data for testing context-DL model
test0 = recommender0(recomm_list, mf)
test1 = recommender1(recomm_list, 37)

# Context variable을 사용한 추천엔진 결합 ##################################################################

# Defining RMSE measure
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

reg = 0.025                                                         # Regularization penalty
recomm0 = Input(shape=(1,))                                         # User input
recomm1 = Input(shape=(1,))                                         # Item input
context1 = Input(shape=(19,))                                       # Movie context (genre)
context2 = Input(shape=(1,))                                        # User context (occupation)
r0_layer = Dense(8)(recomm0)                                        # Recommender 1
r0_layer = Activation('linear')(r0_layer)
r0_layer = Flatten()(r0_layer)
r1_layer = Dense(8)(recomm1)                                        # Recommender 2
r1_layer = Activation('linear')(r1_layer)
r1_layer = Flatten()(r1_layer)

context_layer1 = Dense(8)(context1)
context_layer1 = Activation('gelu')(context_layer1)
context_layer1 = Flatten()(context_layer1)

context_layer2 = Embedding(L,8, embeddings_regularizer=l2(reg))(context2)
context_laeyr2 = Activation('gelu')(context_layer2)
context_layer2 = Flatten()(context_layer2)

interaction0_layer = Multiply()([r0_layer, context_layer1, context_layer2])     #추천값, occupation, genre 싹 다 곱하기
interaction0_layer = Flatten()(interaction0_layer)
interaction1_layer = Multiply()([r1_layer, context_layer1, context_layer2])
interaction1_layer = Flatten()(interaction1_layer)
interaction_layer = Concatenate()([interaction0_layer, interaction1_layer])
interaction_layer = Activation('linear')(interaction_layer)
interaction_layer = Dense(64)(interaction_layer)
interaction_layer = Activation('gelu')(interaction_layer)
interaction_layer = Dropout(0.001)(interaction_layer)
interaction_layer = Dense(16)(interaction_layer)
interaction_layer = Activation('linear')(interaction_layer)
interaction_layer = Flatten()(interaction_layer)


R = Concatenate()([r0_layer, r1_layer])
R = Dense(64)(R)
R = Activation('linear')(R)
R = Flatten()(R)
R = Concatenate()([R, context_layer1, context_layer2, interaction_layer])   # 추천정보, occupation, genre, 관계정보 concat

# Adding more layers
R = Dense(512)(R)
R = Activation('linear')(R)
R = Dropout(0.004)(R)

R = Dense(1)(R)

model = Model(inputs=[recomm0, recomm1, context1, context2], outputs=R)
model.compile(
  loss=RMSE,
  #optimizer=SGD(lr=0.001, momentum=0.95),
  optimizer=Adamax(lr=0.0000001),
  metrics=[RMSE]
)
model.summary()

checkpoint_path = 'CheckPoint'
checkpoint = ModelCheckpoint(checkpoint_path, 
                             save_best_only=True, 
                             save_weights_only=True, 
                             monitor='val_RMSE', 
                             verbose=1)

result = model.fit(
  x=[train0, train1, context_train1, context_train2],
  y=ratings_train.rating.values.astype(np.float64),
  callbacks=[checkpoint],
  epochs=15,
  batch_size=16384,
  validation_data=(
    [test0, test1, context_test1, context_test2],
    ratings_test.rating.values.astype(np.float64)
  )
)

model.load_weights(checkpoint_path)

# Prediction
predictions = np.ravel(model.predict([test0, test1, context_test1, context_test2]), order='C')
predictions[predictions > 5] = 5
predictions[predictions < 1] = 1
print(RMSE2(np.array(ratings_test['rating']), predictions))
print(RMSE2(np.array(ratings_test['rating']), test0))
print(RMSE2(np.array(ratings_test['rating']), test1))


