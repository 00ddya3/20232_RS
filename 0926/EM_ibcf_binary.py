# Created or modified on Sep 2023
# Author: 임일
# EachMovie IBCF binary (precision, recall, F1 구하기)


import numpy as np
import pandas as pd

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=12)
rating_matrix_t = x_train.pivot(values='rating', index='movie_id', columns='user_id')
x_test = x_test.set_index('user_id')
x_train = x_train.set_index('user_id')

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
    
def score_binary(model, n_of_recomm=10, ref_size=2):    #10개를 추천하겠다
    precisions = []
    recalls = []
    F1s = []
    for user in set(x_test.index):              # Test set에 있는 모든 사용자 각각에 대해서 실행
        y_true = np.array(x_test.loc[user]['movie_id'])
        #y_true = x_test.loc[user][x_test.loc[user]['rating'] >= cutline]['movie_id']    # cutline 이상의 rating만 정확한 것으로 간주
        if n_of_recomm == 0:                    # 실제 평가한 영화수 같은 수만큼 추천 
            n_of_recomm = len(y_true)
        y_pred = np.array(model(user, n_of_recomm, ref_size))
        precision = 0
        recall = 0
        F1 = 0
        if (len(y_true) > 0 and len(y_pred) > 0):   # Check if y_true and y_pred > 0
            precision, recall, F1 = b_metrics1(y_true, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
    return np.mean(precisions), np.mean(recalls), np.mean(F1s)

# 아이템 pair의 Cosine similarities 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix_t.copy().fillna(0)
item_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
item_similarity = pd.DataFrame(item_similarity, index=rating_matrix_t.index, columns=rating_matrix_t.index)

def ibcf_binary(user, n_of_recomm=10, ref_size=2):
    rated_index = rating_matrix_t[user][rating_matrix_t[user] > 0].index            # 현 사용자가 rating한 item 저장
    ref_group = rating_matrix_t[user].sort_values(ascending=False)[:ref_size]       # 현 사용자가 가장 높게 평가한 item (ref group) 추출
    sim_scores = item_similarity[ref_group.index].mean(axis=1)                      # 이 item(ref group)들과 유사도 평균 계산
    sim_scores = sim_scores.drop(rated_index)                                       # 이미 평가한 item은 제외
    recommendations = sim_scores.sort_values(ascending=False)[:n_of_recomm].index   # 평균 유사도가 가장 높은 n개 item 추출
    return recommendations

# 정확도 계산
score_binary(ibcf_binary, 22, 11)


