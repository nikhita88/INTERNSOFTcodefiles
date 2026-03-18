# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import modules

import pandas as pd
import matplotlib.pyplot as plt

#reading data from values

data = pd.read_csv("advertising.csv")
data.head()


#to visualise data


fig, axs = plt.subplots(1,3,sharey = True)
data.plot(kind = 'scatter', x = 'TV', y = 'Sales', ax = axs[0], figsize=(12,4))
data.plot(kind = 'scatter', x = 'Radio', y = 'Sales', ax = axs[1])
data.plot(kind = 'scatter', x = 'Newspaper', y = 'Sales', ax = axs[2])

feature_cols = ['TV']

X = data[feature_cols]
y = data.Sales

#importing Linear Regression Algo

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

#y = a+bx



x_new = pd.DataFrame({'TV':[data.TV.min(), data.TV.max()]})
x_new.head()

prediction = lr.predict(x_new)
prediction

data.plot(kind = 'scatter', x='TV', y='Sales')
plt.plot(x_new, prediction, c='red', linewidth = 3)

#summary

import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV', data = data).fit()
lm.conf_int()

#finding probability values
lm.pvalues

#finding the R-SQUARED values
lm.rsquared

#MULTI LINEAR REGRESSION


feature_cols = ['TV', 'Radio', 'Newspaper']

X = data[feature_cols]
y = data.Sales

lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

lm = smf.ols(formula='Sales ~ TV+Radio+Newspaper', data = data).fit()
lm.conf_int()
lm.summary()          

 