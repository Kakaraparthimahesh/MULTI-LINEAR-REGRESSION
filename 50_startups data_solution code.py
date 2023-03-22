# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:55:38 2022

@author: MAHESH
"""
# import the data set

import pandas as pd
df = pd.read_csv("50_Startups.csv")
df.shape
list(df)
df.head()

# Label encoding

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['R&D Spend'] = LE.fit_transform(df['R&D Spend'])
df['Administration'] = LE.fit_transform(df['Administration'])
df['Marketing Spend'] = LE.fit_transform(df['Marketing Spend'])
df['State'] = LE.fit_transform(df['State'])
df['Profit'] = LE.fit_transform(df['Profit'])

df.info()  # list of the variable names with what the data type it is after label encoding.

# finding the average of all all variables

df ["R&D Spend"].describe()
df ["Administration"].describe()
df ["Marketing Spend"].describe()
df ["State"].describe()
df ["Profit"].describe()

# split the variables as x and y

X = df[['R&D Spend','Administration','Marketing Spend','State']]   # MODEL 1

# X = df[['R&D Spend','Marketing Spend']]                          # MODEL 2
 
# X = df[['R&D Spend','Marketing Spend','Administration']]         # MODEL 3

# X = df[['R&D Spend','Marketing Spend','State']]                  # MODEL 4

Y = df['Profit']

df.corr()

# <<<<< EXPLORATION DATA ANALYSIS <<<<<

# kurtosis,skew 

from  scipy.stats import kurtosis,skew

kurtosis(df['R&D Spend'],fisher=False)
kurtosis(df['Administration'],fisher=False)
kurtosis(df['Marketing Spend'],fisher=False)
kurtosis(df['State'],fisher=False)
kurtosis(df['Profit'],fisher=False)

skew(df['R&D Spend'])
skew(df['Administration'])
skew(df['Marketing Spend'])
skew(df['State'])
skew(df['Profit'])

# histograme 

df.hist('R&D Spend')
df.hist('Administration')
df.hist('Marketing Spend')
df.hist('State')
df.hist('Profit')

# box plot 

df.boxplot('R&D Spend')
df.boxplot('Administration')
df.boxplot('Marketing Spend')
df.boxplot('State')
df.boxplot('Profit')

# Scatter plot between the variables along with histograms

import seaborn as sns
sns.pairplot(df)

# columns names
df.columns

# Fitting LinearRegression

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

LR.intercept_    # B0 
LR.coef_         # B1 

#predicating the values

Y_pred = LR.predict(X)

# calculating mean square error

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y, Y_pred)
mse

import numpy as np
RMSE = np.sqrt(mse)
print('Root Mean square error of above models is:', RMSE.round(2))

R2 = r2_score(Y, Y_pred)
print("R square performance of above model is:", (R2*100).round(2))

# How to calculate the VIF value
# split the variables as x and y

# MODEL 1

Y = df['R&D Spend']                       
X = df[['Administration','Marketing Spend','State']]

# MODEL 2

# Y = df['R&D Spend']
# X = df[['Marketing Spend']]

# MODEL 3

# Y = df['R&D Spend']
# X = df[['Marketing Spend','Administration']]

# MODEL 4

# Y = df['R&D Spend']
# X = df[['Marketing Spend','State']] 

LR = LinearRegression()
LR.fit(X, Y)
Y_pred = LR.predict(X)
R2 = r2_score(Y, Y_pred)
vif = 1 / (1-R2)
print("vif is:", vif)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# if the value of vif is less then 5 that is best model
# if the value of vif is between 5 to 10 then that model is ok 
# and that model contains some collinearity issues
# if the value of vif is grater then 10 that model is not good
# and that model contains lot of collinearity issues

#=======================================================================================================

# TYPE             M1                    M2                   M3                   M4 

## MSE  =   3.9562452821769063    4.053939105328656    4.010920438376176    3.9961479027894313

## RMSE =         1.99                  2.01                  2.0                   2.0

## R^2  =        98.1                   98.05                98.07                 98.08

## vif  =  2.350520672967834      2.019029200655878    2.3453744163516257    2.0271532260450233

#=======================================================================================================

# Model 1 is the best model when compared with remaining models

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

