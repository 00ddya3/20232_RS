# Created or modified on Nov 2023
# author: 임일
# Amazon review 데이터로 W2V 사용 

import pandas as pd
reviews = pd.read_csv('C://Recosys/Data/Reviews2.csv', encoding='latin-1')
reviews = reviews['Text']
texts = '\n'.join(reviews[:5000])   # 5000개 리뷰만 읽기

# 특수 문자 제거 예
import re
texts = re.sub("\([^)]*\)", " ", str(texts))

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
sent_text = sent_tokenize(texts)

# 텍스트 normalize
normalized_text = []
for string in sent_text:
     tokens = re.sub("[^a-zA-Z\u3131-\u3163\uac00-\ud7a3]", " ", string.lower())     # ㄱ-ㅣ가-힣
     normalized_text.append(tokens)

# 단어 토큰화
result = [word_tokenize(sentence) for sentence in normalized_text]

from gensim.models import Word2Vec
model = Word2Vec(sentences=result, vector_size=500, window=10, min_count=5, workers=4, sg=1)

model_result = model.wv.most_similar("sweet", topn=5)
print(model_result)

from gensim.models import KeyedVectors
model.wv.save_word2vec_format('eng_w2v')                    # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드
model_result = loaded_model.most_similar("sweet", topn=20)
print(model_result)

