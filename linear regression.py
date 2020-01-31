# -*- coding: utf-8 -*-
#import libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


#data processing
data=pd.read_csv('Salary_Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


#train_split_test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


#linear regression
from sklearn.linear_model import LinearRegression
linearregression=LinearRegression()
linearregression.fit(x_train,y_train)


#predict the model
y_pred=linearregression.predict(x_test)


#plot the result 
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,linearregression.predict(x_train),color='blue')
plt.tittle('linear regression')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

