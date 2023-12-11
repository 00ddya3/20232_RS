# Created or modified on Nov 2023
# Author: 임일
# Sparse matrix 1

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ratings = {'user_id': [1,2,4], 
     'movie_id': [2,3,7], 
     'rating': [4,3,1]}
ratings = pd.DataFrame(ratings)

# Pandas pivot을 이용해서 full matrix로 변환하는 경우
ratings_matrix = ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)
print(ratings_matrix)
np.array(ratings_matrix)

# Sparse matrix를 이용해서 full matrix로 변환하는 경우
data = np.array(ratings['rating'])
row_indices = np.array(ratings['user_id'])
col_indices = np.array(ratings['movie_id'])
ratings_matrix = csr_matrix((data, (row_indices, col_indices)), dtype=int)
print(ratings_matrix)
ratings_matrix.toarray()

