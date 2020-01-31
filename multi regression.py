# -*- coding: utf-8 -*-

#import libraries
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


#data processing
data=pd.read_csv('50_Startups.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,4].values

#categoricall 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder 
labelecnoder=LabelEncoder()
x[:,3]=labelecnoder.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()


#to avoid Dymmy trap
x=x[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)




#multi linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


#predict
y_pred=regressor.predict(x_test)

#**x-y must be the same size so it wont plot**
#plot the result 
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.tittle('multi regression  model')
plt.xlabel('state-years of experience')
plt.ylabel('profit')
plt.show()