# Created or modified on Sep 2023
# Author: 임일
# UBCF bias-from-mean binary (precision, recall, F1 구하기)

import pandas as pd
import numpy as np

# Read rating data
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings.drop('timestamp', axis=1)

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=12)
rating_matrix = x_train.pivot(values='rating', index='user_id', columns='movie_id')
x_test = x_test.set_index('user_id')
x_train = x_train.set_index('user_id')

# 모든 가능한 사용자 pair의 Cosine similarities 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# 모든 user의 rating 평균 계산 
rating_mean = rating_matrix.mean(axis=1)

# 사용자의 평가경향을 고려한 추천
def ubcf_bias(user_id, movie_id):
    import numpy as np
    # 현 user의 평균 가져오기
    user_mean = rating_mean[user_id]
    
    if movie_id in rating_matrix:
        # 현 user와 다른 사용자의 유사도 가져오기
        sim_scores = user_similarity[user_id]
        # 현 movie의 모든 rating 가져오기
        movie_ratings = rating_matrix[movie_id]
        # 모든 사용자의 rating 평균 가져오기
        others_mean = rating_mean
        # 현 movie에 대한 rating이 없는 user 삭제
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        others_mean = others_mean.drop(none_rating_idx)
        # 편차 예측치 계산
        movie_ratings = movie_ratings - others_mean
        if sim_scores.sum() > 0.000001:
            prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum() + user_mean
        else:    
            prediction = user_mean
    else:
        prediction = user_mean
    return prediction

# precision, recall, F1 계산을 위한 함수
def b_metrics1(target, pred):       # 실제, 예측 item을 리스트로 받아서 precision, recall, F1 계산하는 함수
    n_target = len(target)          # item 개수 초기화
    n_pred = len(pred)
    n_correct = len(set(target).intersection(set(pred)))
    try:                            # 에러(division by zero 등)가 발생하는 경우를 대비해서
        precision = n_correct / n_pred
        recall = n_correct / n_target
        if (precision == 0 and recall == 0):  # Prevent 'division by zero'
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1
    except:
        return 'error'

def score_binary(model, n_of_recomm=10, cutline=3):
    precisions = []
    recalls = []
    F1s = []
    for user in set(x_test.index):              # Test set에 있는 모든 사용자 각각에 대해서 실행
        #y_true = x_test.loc[user]['movie_id']
        y_true = x_test.loc[user][x_test.loc[user]['rating'] >= cutline]['movie_id'] # cutline 이상의 rating만 정확한 것으로 간주
        if n_of_recomm == 0:                    # 실제 평가한 영화수 같은 수만큼 추천 
            n_of_recomm = len(y_true)
        y_pred = model(user, n_of_recomm)
        precision, recall, F1 = b_metrics1(y_true, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
    return np.mean(precisions), np.mean(recalls), np.mean(F1s)

# 원래 function을 이용하는 방법 (느림)
def ubcf_binary(user, n_of_recomm=10):
    items = rating_matrix.loc[user]                 # 현 사용자의 rating
    rated_index = items[items > 0].index            # 현 사용자가 평가한 item 확인
    items = items.drop(rated_index)                 # 현 사용자가 평가한 item을 제외
    predictions = []
    for item in items.index:                        
        predictions.append(ubcf_bias(user, item))   # 평가하지 않은 모든 item에 대한 예상 평점 계산
    predictions = pd.Series(data=predictions, index=items.index)
    recommendations = predictions.sort_values(ascending=False)[:n_of_recomm].index  # 상위 n개의 id 반환
    return recommendations

score_binary(ubcf_binary, 20)




# Matrix 연산을 사용하는 방법 (빠름)
rating_matrix_bias = rating_matrix.sub(rating_mean, axis=0)     # 평균을 뺀다 (bias 계산)
rated_index = (rating_matrix_bias.notna())                      # rating이 있는 element 표시
rating_matrix_bias = rating_matrix_bias.fillna(0)               # 연산을 위해 null을 0으로 변경
sum_sim_scores = np.array(user_similarity).dot(rated_index)     # 현 아이템을 rating한 사용자와 현 사용자의 유사도 합계
sum_sim_scores = pd.DataFrame(data=sum_sim_scores, index=rating_matrix_bias.index, columns=rating_matrix_bias.columns)
sum_sim_scores[sum_sim_scores < 0.000001 ] = 1                  # 'division by zero' 방지

def ubcf_binary2(user, n_of_recomm=10):                                 # matrix로 계산하는 ubcf
    user_rated = rated_index.loc[user][rated_index.loc[user]].index     # 현 사용자가 이미 rating한 item 확인
    # CF(사용자간 유사도로 각 사용자의 rating을 가중평균) 방식으로 모든 item에 대한 현 사용자의 예상평점 계산
    predictions = np.array(user_similarity[user]).dot(np.array(rating_matrix_bias)) / np.array(sum_sim_scores.loc[user])
    predictions = pd.Series(data=predictions, index=rating_matrix_bias.columns)
    predictions = predictions.drop(user_rated)                          # 현 사용자가 평가한 item 제외
    recommendations = predictions.sort_values(ascending=False)[:n_of_recomm].index  # 상위 n개의 id 반환
    return recommendations

score_binary(ubcf_binary2, 20)



