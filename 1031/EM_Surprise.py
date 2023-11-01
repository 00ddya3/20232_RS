import numpy as np
import pandas as pd

# Importing algorithms from Surprise
from surprise import SVD
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import SVDpp
from surprise import NMF

# Importing other modules from Surprise
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)


# 알고리즘 비교
trainset, testset = train_test_split(data, test_size=0.25)
algorithms = [KNNWithMeans, KNNWithZScore, SVD, SVDpp, NMF]   #비교할 알고리즘 리스트

# 결과를 저장할 변수 
names = []
results = []
# Loop 
for option in algorithms:
    algo = option()
    names.append(option.__name__)       # 알고리즘 이름 
    algo.fit(trainset)
    predictions = algo.test(testset)
    results.append(accuracy.rmse(predictions))
names = np.array(names)
results = np.array(results)

# 결과를 그래프로 표시
import matplotlib.pyplot as plt

index = np.argsort(results)
plt.ylim(0.8,1)
plt.plot(names[index], results[index])
