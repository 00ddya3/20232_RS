# Created or modified on Oct 2023
# Author: 임일
# Surprise 1

# Surprise 설치
# !pip install scikit-surprise

import pandas as pd

# Importing algorithms from Surprise
from surprise import KNNWithMeans
from surprise import BaselineOnly

# Importing other modules from Surprise
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

# Importing built in MovieLens 100K dataset
data = Dataset.load_builtin('ml-100k')

# Baseline 알고리즘 지정
algo = BaselineOnly()
# cv=4는 데이터를 4개로 나누어서 하나를 test set으로 사용하는데 5개 모두에 대해서 실행
result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=4, verbose=True)

# Set full train data 지정, 예측하기 
trainset = data.build_full_trainset()
pred = algo.predict('1', '2', r_ui=3, verbose=True)  # user_id, item_id, default rating ->유저1 아이템2에 대한 예측값

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=4, verbose=True)

# Train/Test 분리 계산 
trainset, testset = train_test_split(data, test_size=0.25)
algo = KNNWithMeans()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Neighbor size 변경
trainset, testset = train_test_split(data, test_size=0.25)
algo = KNNWithMeans(k=70)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# 다양한 Neighbor size 비교 
trainset, testset = train_test_split(data, test_size=0.25)
for neighbor_size in (30, 40, 50, 60, 70, 80):
    algo = KNNWithMeans(k=neighbor_size)
    algo.fit(trainset)
    predictions = algo.test(testset)
    print('K = ', neighbor_size, 'RMSE = ', accuracy.rmse(predictions, verbose=False))

# 알고리즘 옵션 변경
trainset, testset = train_test_split(data, test_size=0.25)
sim_options = {'name': 'cosine',    # 유사도 측정 옵션
               'user_based': True   # false면 item-based
               }
algo = KNNWithMeans(k=70, sim_options=sim_options)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
