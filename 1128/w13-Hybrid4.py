# Created or modified on Nov 2023
# Author: 임일
# Hybrid 4 (CF + MF, Binary)

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Read rating data
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies[['movie_id', 'title']]
movies = movies.set_index('movie_id')
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)

# train test 분리
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state=12)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]
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

# IBCF 정확도 계산  ############################################################################
# 아이템 pair의 Cosine similarities 계산
from sklearn.metrics.pairwise import cosine_similarity
ratings_train_t = ratings_train.pivot(values='rating', index='movie_id', columns='user_id')
ratings_test = ratings_test.set_index('user_id')

matrix_dummy = ratings_train_t.copy().fillna(0)
item_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
item_similarity = pd.DataFrame(item_similarity, index=ratings_train_t.index, columns=ratings_train_t.index)

# precision, recall, F1 계산을 위한 함수
def b_metrics(y_true, y_pred):
    try:
        n_match = set(y_true).intersection(set(y_pred))
        precision = len(n_match) / len(y_pred)
        recall = len(n_match) / len(y_true)
        F1 = 2 * (precision * recall) / (precision + recall)
    except:
        precision = 0
        recall = 0
        F1 = 0
    return precision, recall, F1

def ibcf_binary(user, n_of_recomm=10, ref_size=2):
    rated_index = ratings_train_t[user][ratings_train_t[user] > 0].index
    ref_group = ratings_train_t[user].sort_values(ascending=False)[:ref_size]
    sim_scores = item_similarity[ref_group.index].mean(axis=1)
    sim_scores = sim_scores.drop(rated_index)
    recommendations = pd.Series(sim_scores.sort_values(ascending=False)[:n_of_recomm].index)
    return recommendations

def score_binary(model, n_of_recomm=10, ref_size=2):
    precisions = []
    recalls = []
    F1s = []
    for user in set(ratings_test.index):            # Test set에 있는 모든 사용자 각각에 대해서 실행
        y_true = ratings_test.loc[user]['movie_id']
        if n_of_recomm == 0:                        # 실제 평가한 영화수 같은 수만큼 추천 
            n_of_recomm = len(y_true)
        y_pred = model(user, n_of_recomm, ref_size)
        precision, recall, F1 = b_metrics(y_true, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
    return np.mean(precisions), np.mean(recalls), np.mean(F1s)

# IBCF 정확도 계산
print('Original IBCF: ', score_binary(ibcf_binary, 0, 15))


# Combine IBCF and MF ##############################################################
def score_hybrid(n_of_candidates=50, n_of_recomm=10, ref_size=2):
    precisions = []
    recalls = []
    F1s = []
    for user in set(ratings_test.index):            # Test set에 있는 모든 사용자 각각에 대해서 실행
        y_true = ratings_test.loc[user]['movie_id']
        if n_of_recomm == 0:                        # 실제 평가한 영화수 같은 수만큼 추천 
            n_of_recomm = len(y_true)
        y_pred = []
        items = ibcf_binary(user, n_of_candidates, ref_size)
        for item in items:
            y_pred.append(mf.get_one_prediction(user, item))
        y_pred = pd.Series(y_pred, index=items)
        recommendations = y_pred.sort_values(ascending=False)[:n_of_recomm]
        recommendations = recommendations.index
        precision, recall, F1 = b_metrics(y_true, recommendations)
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
    return np.mean(precisions), np.mean(recalls), np.mean(F1s)

score_hybrid(30, 30, 15)

# Hybrid 정확도 계산
for i in range(25, 35):
    print('Hybrid: ',i, score_hybrid(i, 0, 15))


