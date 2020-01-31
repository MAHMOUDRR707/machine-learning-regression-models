# -*- coding: utf-8 -*-

#imoporting libraries 
import matplotlib.pyplot as plt
import  numpy as np 
import pandas as pd



#data processing
data=pd.read_csv('50_Startups.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,4].values


#categorical scalling
from sklearn.preprocessing  import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()


#avoid dummy trap 
x=x[:,1:]

#split_test_train
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=0)



#multi linear regression
from  sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)


#plot the result
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.tittle('startup')
plt.xlabel('independent varaibles')
plt.ylabel('profits')


#backwards emlination
import statsmodels.api as sm
x=np.append(np.ones((50,1)).astype(int),x,axis=1)
x_opt=x[:,[0,1,2,3,4]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

