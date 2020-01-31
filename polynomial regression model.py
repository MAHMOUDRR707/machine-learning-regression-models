# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)


#plot ploniminal 
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#polynomial  model
from sklearn.preprocessing import PolynomialFeatures 
regressor=PolynomialFeatures(degree=4)
x_poly=regressor.fit_transform(x)
regressor.fit_transform(x_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)


#plot ploynomial
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg2.predict(regressor.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


