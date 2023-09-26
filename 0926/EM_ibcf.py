# Created or modified on Sep 2023
# Author: 임일
# EachMovie IBCF

import numpy as np
import pandas as pd

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')

from sklearn.model_selection import train_test_split

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
x = ratings.copy()
y = ratings['user_id'] 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=12)
rating_matrix_t = x_train.pivot(values='rating', index='movie_id', columns='user_id')

# RMSE 계산을 위한 함수
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def score(model):
    id_pairs = zip(x_test['user_id'], x_test['movie_id'])
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
    y_true = np.array(x_test['rating'])
    return RMSE(y_true, y_pred)







def ibcf(user_id, movie_id):

    




score(ibcf)

