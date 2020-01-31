# -*- coding: utf-8 -*-

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#splitingg testing  training the data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#decision tree model
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=15,random_state=0)
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title('random forest regressor model')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

