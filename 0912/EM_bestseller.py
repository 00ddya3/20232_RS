# Created or modifed on Sep 2022
# Author: 임일
# w2 EachMovie연습문제

import pandas as pd

# csv 파일에서 불러오기
ratings = pd.read_csv('C:/RecoSys/Data/EM_ratings.csv', encoding='utf-8')
ratings = ratings.set_index('user_id')

movies = pd.read_csv('C:/RecoSys/Data/movie.csv', encoding='utf-8')
