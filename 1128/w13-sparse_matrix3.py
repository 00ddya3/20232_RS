# Created or modified on Nov 2023
# Author: 임일
# Sparse matrix 3

import numpy as np
import pandas as pd
import datetime
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle

# 데이터 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
#ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')         # 100K data
#ratings = pd.read_csv('C:/RecoSys/Data/ratings-1m.csv', names=r_cols,  sep=',',encoding='latin-1')  # 1M data
ratings = pd.read_csv('C:/RecoSys/Data/ratings-20m.csv', names=r_cols,  sep=',',encoding='latin-1') # 20M data
ratings = ratings.drop('timestamp', axis=1)

# train test 분리
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state=12)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

# Sparse matrix 사용
data = np.array(ratings['rating'])
row_indices = np.array(ratings['user_id'])
col_indices = np.array(ratings['movie_id'])
ratings = csr_matrix((data, (row_indices, col_indices)), dtype=int)

# New MF class for training & testing
class NEW_MF():
    # Initializing the object
    def __init__(self, ratings, K, alpha, beta, iterations, tolerance=0.005, verbose=True):
        self.R = ratings
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
            x = ratings_test.iloc[i,0]                          # Getting R indice for the given user_id and movie_id
            y = ratings_test.iloc[i,1]
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
        start = datetime.datetime.now()
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 1 == 0:
                    stop = datetime.datetime.now()
                    duration = stop - start
                    start = datetime.datetime.now()
                    print("Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %4f" % (i+1, rmse1, rmse2))
                    print("Time duration: %d.%d seconds" % (duration.seconds, duration.microseconds))
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

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

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
        return self.get_prediction(user_id, movie_id)

# Testing MF RMSE
R_temp = ratings.copy()          # Save original data
mf = NEW_MF(R_temp, K=105, alpha=0.001, beta=0.014, iterations=200, tolerance=0.001, verbose=True)
test_set = mf.set_test(ratings_test)
result = mf.test()









