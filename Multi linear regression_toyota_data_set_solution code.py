# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 20:48:08 2022

@author: MAHESH
"""
# import the data set

import pandas as pd
pd.set_option("display.max_columns", 20)
df = pd.read_csv("ToyotaCorolla.csv",encoding = 'latin1')
df.shape
list(df)
df.columns
df.info()             # list of the variable names with data set
df.isnull().sum()     # finding missing values

# Label encoding

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df.iloc[:,5]

for eachcolumn in range(0,37):
    df.iloc[:,eachcolumn] = LE.fit_transform(df.iloc[:,eachcolumn])

df.head()

# droping the variables

df.drop(['Mfg_Month','Mfg_Year','Fuel_Type','Met_Color','Automatic','Cylinders','Mfr_Guarantee','BOVAG_Guarantee','Guarantee_Period','ABS','Airbag_1','Airbag_2','Airco','Automatic_airco','Boardcomputer','CD_Player','Central_Lock','Powered_Windows','Power_Steering','Radio','Mistlamps','Sport_Model','Backseat_Divider','Metallic_Rim','Radio_cassette','Tow_Bar'], axis=1,inplace=True)
list(df)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  or <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# WITH OUT DROPPING WE CAN MAKE CREATE A NEW DARA SET LIKE THIS AS SHOWN IN THE BELOW LINE 36 
# df_new = df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# split the variables as x and y

X = df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]      # M1

# X = df[["Age_08_04","KM","Weight"]]                                              # M2

# X = df[["Age_08_04","KM","Doors","Weight"]]                                      # M3

# X = df[["Age_08_04","KM"]]                                                       # M4                

# X = df[["HP","Doors","Gears","Weight"]]                                          # M5

# X = df[["Age_08_04","KM","HP","Quarterly_Tax","Weight"]]                         # M6

Y = df["Price"]

# <<<<< EXPLORATION DATA ANALYSIS <<<<<

# kurtosis,skew 

from  scipy.stats import kurtosis,skew

kurtosis(df['Age_08_04'],fisher=False)
kurtosis(df['KM'],fisher=False)
kurtosis(df['HP'],fisher=False)
kurtosis(df['cc'],fisher=False)
kurtosis(df['Doors'],fisher=False)
kurtosis(df['Gears'],fisher=False)
kurtosis(df['Quarterly_Tax'],fisher=False)
kurtosis(df['Weight'],fisher=False)

skew(df['Age_08_04'])
skew(df['KM'])
skew(df['HP'])
skew(df['cc'])
skew(df['Doors'])
skew(df['Gears'])
skew(df['Quarterly_Tax'])
skew(df['Weight'])

# histograme 

df.hist("Age_08_04")
df.hist("KM")
df.hist("HP")
df.hist("cc")
df.hist("Doors")
df.hist("Gears")
df.hist("Quarterly_Tax")
df.hist("Weight")
df.hist("Price")

# box plot 

df.boxplot("Age_08_04")
df.boxplot("KM")
df.boxplot("HP")
df.boxplot("cc")
df.boxplot("Doors")
df.boxplot("Gears")
df.boxplot("Quarterly_Tax")
df.boxplot("Weight")
df.boxplot("Price")

# Scatter plot between the variables along with histograms

import seaborn as sns
sns.pairplot(df)

# scattre plot

df.plot.scatter(x='Age_08_04', y='Price',color='red')          
df.plot.scatter(x='KM', y='Price',color='yellow') 
df.plot.scatter(x='HP', y='Price',color='pink')    
df.plot.scatter(x='cc', y='Price',color='blue')
df.plot.scatter(x='Doors', y='Price',color='green')
df.plot.scatter(x='Gears', y='Price',color='black')
df.plot.scatter(x='Quarterly_Tax', y='Price',color='orange')
df.plot.scatter(x='Weight', y='Price',color='skyblue')

# columns names
df.columns

df.corr()

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

# -------------------------------------------
# How to calculate the VIF value

# split the variables as x and y

# MODEL 1
Y = df[["Age_08_04"]]                                                  # M1                       
X = df[["KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

# MODEL 2 
# Y = df[["Age_08_04"]]                                                # M2
# X = df[["KM","Weight"]]                                              

# MODEL 3
# Y = df[["Doors"]]                                                    # M3
# X = df[["Age_08_04","KM","Weight"]]

# MODEL 4
# Y = df[["KM"]]                                                       # M4
# X = df[["Age_08_04"]]                                                    

# MODEL 5
# Y = df[["HP"]]                                                       # M5
# X = df[["Doors","Gears","Weight"]]                                         

# MODEL 6
# Y = df[["KM"]]                                                       # M6
# X = df[["Age_08_04","HP","Quarterly_Tax","Weight"]]

LR = LinearRegression()
LR.fit(X, Y)
Y_pred = LR.predict(X)
R2 = r2_score(Y, Y_pred)
vif = 1 / (1-R2)
print("vif is:", vif)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# TYPE             M1                 M2                  M3                   M4                 M5                    M6

# MSE  =  447.9373435836156   523.9402838692575   516.4810253102331   551.7753013471879    1893.7423401022004    467.4140298440879
 
# RMSE =       21.16               22.89                22.73               23.49                43.52                 21.62

# R^2  =       83.6                80.82                81.09               79.8                 30.67                 82.89

# vif  =  2.207983037829943   2.148529173903026   1.118193900541456   1.4694087084021212   1.0487979404810575    1.730987793049273

#====================================================================================================================================

# Model 1 is the best model when compared with remaining models

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>






