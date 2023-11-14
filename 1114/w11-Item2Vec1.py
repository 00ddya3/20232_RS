# Created or modified on Nov 2023
# author: 임일
# MovieLens 데이터로 Item2Vec 구현

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Read rating data
r_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('C:/RecoSys/Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')         # 100K data
#ratings = pd.read_csv('C:/RecoSys/Data/ratings-1m.csv', names=r_cols,  sep=',',encoding='latin-1')  # 1M data
ratings = ratings.set_index('user_id')
users = list(set(ratings.index))        # unique user_id 뽑아내기 / 다른 알고리즘과의 차이점
N_ITEM = len(set(ratings['item_id']))

# Item_id를 str으로 변환 (Word2Vec을 이용하기 위함)
ratings['item_id'] = ratings['item_id'].apply(lambda x: str(x))

# train test 분리
TRAIN_SIZE = 0.75       # Train set의 비율
users = shuffle(users, random_state=1)
cutoff = int(TRAIN_SIZE * len(users))
train_users = users[:cutoff]
test_users = users[cutoff:]

# 각 user별 리뷰한 영화를 리스트로 만들고 리뷰한 시간을 기준으로 정렬
train_data = []
for user in train_users:
    train_data.append(ratings.loc[user])
for i in range(len(train_data)):                 # 리뷰한 시간 기준을 정렬하고 item리스트 생성
    train_data[i] = train_data[i].sort_values(by=['timestamp'])
    train_data[i] = list(train_data[i]['item_id'])

# 위 작업을 단순화 한 것
'''train_data = []
for user in train_users:
    train_data.append(list(ratings.loc[user].sort_values(by=['timestamp'])['item_id']))
'''

# Item2Vec을 위한 파라메터
HOLD_SIZE = 20          # Target 아이템의 수
LOOKUP_SIZE = 6         # 몇개의 아이템을 기준으로 추천할 것인
WINDOW_SIZE = 20        # item당 몇개의 유사 item을 선정할 것인가

# test data 준비
test_data = []
for user in test_users:
    if len(ratings.loc[user]) > HOLD_SIZE + LOOKUP_SIZE:  # 추천 item을 만들기 충분한 숫자인지 확인
        user_items = list(ratings.loc[user].sort_values(by=['timestamp'])['item_id'])
        test_data.append([user_items[:-HOLD_SIZE], user_items[-HOLD_SIZE:]])

# Item2Vec
from gensim.models import Word2Vec
model = Word2Vec(sentences=train_data, vector_size=N_ITEM, window=WINDOW_SIZE, min_count=5, workers=4, sg=1)

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
    for test_user in test_data:
        for item in test_user[0][-LOOKUP_SIZE:]:
            try:            #  model에서 error가 생기는 경우에 대비
                model_result = model.wv.most_similar(item, topn=60)
                recomm_list = {}
                for result in model_result:
                    if result[0] not in set(test_user[0]):       # 사용자가 이미 평가한 것은 제외
                        if result[0] not in recomm_list:         # 리스트에 없는 item이면 리스트에 추가
                            recomm_list[result[0]] = []
                        recomm_list[result[0]].append(result[1]) # 기존 item에 유사도를 추가
            except:         # 이 item은 추천 불가이므로 pass
                pass
        for item in recomm_list:                                 # 여러번 추천된 아이템의 유사도 평균값 계산
            recomm_list[item] = np.mean(recomm_list[item])
        recomm_list = sorted(recomm_list.items(), key=lambda x: x[1], reverse = True)   # 유사도값으로 정렬
        recomm_list = np.array(recomm_list)[:recomm_size, 0].astype(str)                # 추천 item 수만큼 id 받기
        p, r, f = b_metrics(test_user[1], recomm_list)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

score(20)

for size in range(10, 70, 10):
    print('Size = ', size, score(size))


