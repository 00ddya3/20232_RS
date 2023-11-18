# Created or modified on Nov 2023
# Author: 임일
# Graph RA

# Import libraries
import pandas as pd
import numpy as np
import time

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)            # timestamp 제거

# Load the u.item file into a dataframe
import pandas as pd
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies[['movie_id', 'title']]

# train test 분리
TRAIN_SIZE = 0.75
from sklearn.utils import shuffle
ratings = shuffle(ratings, random_state=12)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]
rating_matrix = ratings_train.pivot(values='rating', index='user_id', columns='movie_id')

# 사용자, 영화 일련번호 부여하기    user id랑 numpy index가 달라서 수행함
user_idx = {}
movie_idx = {}
idx_user = {}
idx_movie = {}
for i in set(ratings.user_id):
    user_idx[i] = len(user_idx)
for i in set(ratings.movie_id):
    movie_idx[i] = len(movie_idx)
for i in user_idx:
    idx_user[user_idx[i]] = i
for i in movie_idx:
    idx_movie[movie_idx[i]] = i

M = len(user_idx)   # Number of users
N = len(movie_idx)  # Number of movies


# numpy array로 변환 (item X user)
train_data = np.zeros((N, M))
for rating in np.array(ratings_train):
    train_data[movie_idx[rating[1]], user_idx[rating[0]]] = rating[2]
train_data[train_data > 0] = 1

# 공통 구매 횟수 계산
# for loop 사용
start = time.time()
times_purchased = np.zeros((N, N))
for i in range(N):
    for j in range(i, N):
        for k in range(M):
            times_purchased[i,j] += train_data[i,k] and train_data[j,k]
        times_purchased[j,i] = times_purchased[i,j]     # 대칭을 채워주기 위함
end = time.time()
print(f"{end - start:.5f} sec")

# product * product matrix 사용
start = time.time()
train1 = train_data[np.newaxis, :, :]
train2 = train_data[:, np.newaxis, :]
times_purchased = (train1 * train2).sum(axis=2)
end = time.time()
print(f"{end - start:.5f} sec")


items_sum = times_purchased.sum(axis=0) + 0.0001   # 0.0001은 division by zero 방지용
items_weighted = np.zeros((N, N))
items_weighted = times_purchased / items_sum


# Prepare recommendation
MAX_RECOMM = 150                            # 최대 추천 수
recomm_list = []
for i in range(N):                          # 각 reference 아이템에 대해서
    a_list = items_weighted[i]
    idx = np.argsort(a_list)                # Co-purchase에 따라 정렬
    idx = idx[-MAX_RECOMM-1:-1][-1::-1]     # 필요한 개수만 잘라냄
    index = []
    for j in idx:                           # 원래 id로 변환
        index.append(idx_movie[j])
    recomm_list.append([index, items_weighted[i, idx]])   # 추천 아이템 index와 weight 저장


def b_metrics(target, pred):        # 실제, 예측 item을 리스트로 받아서 precision, recall, F1 계산하는 함수
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

def score(recomm_size):
    precisions = []
    recalls = []
    f1s = []
    for user in set(ratings_test.index):
        reference = ratings_train.loc[user].sort_values(by='rating', ascending=False).iloc[0]['movie_id']   # 기준이 되는 아이템
        a_recomm = recomm_list[reference][0]                # 추천 리스트 가져오기
        a_recomm = set(a_recomm).difference(set(ratings_train.loc[user]['movie_id'])) # train set과 겹치는 것 빼기
        a_recomm = list(a_recomm)[:recomm_size]             # 추천 개수만큼 잘라냄
        a_target = list(ratings_test.loc[user]['movie_id'])
        p, r, f = b_metrics(a_target, a_recomm)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

ratings_train = ratings_train.set_index('user_id')
ratings_test = ratings_test.set_index('user_id')
score(50)

for size in range(50, 150, 10):
    print('Size = ', size, score(size))

