# Created or modified on Sep 2023
# Author: 임일
# EachMovie Bias-from-mean + KNN CF

import numpy as np
import pandas as pd

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=12)
rating_matrix = x_train.pivot(values='rating', index='user_id', columns='movie_id')

# RMSE 계산을 위한 함수
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))




# KNN + 사용자 평가 경향 추천
def ubcf_bias_knn(user_id, movie_id, neighbor_size=20):

    



score(ubcf_bias_knn, 30)


