import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd

columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration',
'model year','origin','car name']
cars = pd.read_table('auto_mpg.data',delim_whitespace=True,names=columns)
print(cars.head())
# print(cars['mpg'])
# cars[["weight"]] = preprocessing.scale(cars[["weight"]])
# cars["mpg"] = preprocessing.scale(cars["mpg"])
lr = LinearRegression()
lr.fit(cars[['weight']], cars['mpg'])
predictions = lr.predict(cars[['weight']])
print(predictions[0:5])
print(cars['mpg'].head())
mse = mean_squared_error(cars['mpg'], predictions)
print(mse)
rmse = mse**(0.5)
print(rmse)