# -*- coding: utf-8 -*-

#importing laibries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot  as plt


#data processing
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


#svm model
from sklearn.svm import SVR
regressor=SVR()
regressor.fit(x_train,y_train )


y_pred=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.show()