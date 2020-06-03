import sklearn.datasets as datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
Boston = datasets.load_boston()
x = Boston.data[:,5]
x = x.reshape(-1,1)
y = Boston.target
y = y.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state =
0)
regr = LinearRegression()
regr.fit(x_train,y_train)
y_pred = regr.predict(x_test)
mse_test = np.sum((y_pred-y_test)**2)/len(y_test)
mae_test = np.sum(np.absolute(y_pred-y_test))/len(y_test)
rmse_test = mse_test ** 0.5
r2_score = 1- (mse_test/ np.var(y_test))
print('Calculate the MSE MAE RMSE：')
print('MSE:{},MAE:{},\nRMSE:{},R2:{}'.format(mse_test,mae_test,rmse_test,r2_score))
print()
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score #R square
mse_test1 = mean_squared_error(y_test,y_pred)
mae_test1 = mean_absolute_error(y_test,y_pred)
rmse_test1 = mse_test1 ** 0.5
r2_score1 = r2_score(y_test,y_pred)
print('MSE,MAE,RMSE using function：')
print('MSE:{},MAE:{},\nRMSE:{},R2:{}'.format(mse_test1,mae_test1,rmse_test1,r2_score1))