# -*- coding: utf-8 -*-

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Data processing
data=pd.read_csv('Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,3].values


#missing data 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='median',axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3]=imputer.transform(x[:, 1:3])


#train_test_split data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=0)


#categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
labelencoder=LabelEncoder()
x[:,0]=labelencoder.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
y=labelencoder.fit_transform(y)


#feature scalling data 
from sklearn.preprocessing import StandardScaler
standardscaler_x=StandardScaler()
x_train=standardscaler_x.fit_transform(x_train)
x_test=standardscaler_x.fit_transform(x_test)
standardscaler_y=StandardScaler()
y=standardscaler_y.fit_transform(y)