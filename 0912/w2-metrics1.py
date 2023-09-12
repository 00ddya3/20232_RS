# Created or modifed on Sep 2023
# @author: 임일
# Metrics

# MAE/MAD 계산
def mad(y_true, y_pred):
    import numpy as np
    return np.mean(abs(np.array(y_true) - np.array(y_pred)))
    
predictions = [1,0,3,4]
targets = [1,2,2,4]
mad = mad(targets, predictions)
print(mad)

# sklearn으로 MSE 계산
from sklearn.metrics import mean_squared_error
import numpy as np
predictions = [1,0,3,4]
targets = [1,2,2,4]
mse = mean_squared_error(targets, predictions)
print(mse)

# RMSE 계산
rmse = np.sqrt(mse)
print(rmse)

# 직접 RMSE 계산
def RMSE(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

predictions = [1,0,3,4]
targets = [1,2,2,4]
rmse = RMSE(targets, predictions)
print(rmse)








