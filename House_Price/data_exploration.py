# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:24:30 2019

@author: striguei
"""

# to handle datasets
import pandas as pd
import numpy as np

import data_management as dm

# for plotting
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# for tree binarisation
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


# to build the models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# to evaluate the models
from sklearn.metrics import mean_squared_error
from math import sqrt

pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

#%%

"""
Load Data Sets
"""

data = dm.load_dataset("train.csv")
print(data.shape)
data.head()

#%%

submission = dm.load_dataset("test.csv")
print(submission.shape)
submission.head()

#%%

"""
Types of Variables
"""

data.dtypes

#%%

print('Number of House Id labels: ', len(data.Id.unique()))
print('Number of Houses in the Dataset: ', len(data))

"""
House ID is unique to every house and thus isn't a useful variable.
"""

#%%

# find categorical variables
categorical = [ var for var in data.columns if data[var].dtype == 'O']
print(' There are {} categorical variables'.format(len(categorical)))

#%%

"""
 There are a few variables that list the year something was 
 built or renovated. We will now capture these variables and
 consider them separately
"""

# Make a list of the numerical variables first
numerical = [var for var in data.columns if data[var].dtype != 'O']

# list of variables that contain year information
year_vars = [var for var in numerical if 'Yr' in var or 'Year' in var]

#%%

data[year_vars].head()

#%%

data.groupby('MoSold')['SalePrice'].median().plot()
plt.title('House price variation in the year')
plt.ylabel('mean House price')

#%%

"""
Let's identify the discrete variables and separate them out from
the rest of the numerical variables
"""

discrete = []
for var in numerical:
    if len( data[var].unique()) < 20 and var not in year_vars:
        print( var, ' values: ', data[var].unique())
        discrete.append(var)
        
print()
print('There are {} discrete variables'.format(len(discrete)))

#%%

"""
Now we're only going to store the the continuous variables in
the numerical variables. Also we're going to remove 'Id' and 'SalePrice'
"""

numerical = [ var for var in numerical if var not in discrete and var not in ['Id', 'SalePrice'] and var not in year_vars ]
print('There are {} numerical and continuous variables'.format( len(numerical)) )

#%%

"""
Let's take a look at which variables have missing values and
what percentage of values are missing.
"""

for var in data.columns:
    if data[var].isnull().sum()>0:
        print(var, data[var].isnull().mean())
        
#%%
"""       
Let's create a list will tell us how many variables we 
have with missing information.
"""

vars_with_na = [var for var in data.columns if data[var].isnull().sum()>0]
print('Total variables that contain missing information: ', len(vars_with_na))


#%%

"""
Let's take a look at the type of variables where a lot of
values are missing.
"""

for var in data.columns:
    if data[var].isnull().mean()>0.80:
        print(var, data[var].unique())
        
#%%
"""
let's now viauslize the outliers in the continuous variables
and also make historgrams to visualize the distributions.
"""
        
for var in numerical: 
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    fig = data.boxplot(column=var)
    fig.set_title('')
    fig.set_ylabel(var)
    
    plt.subplot(1,2,2)
    fig = data[var].hist(bins=20)
    fig.set_ylabel('Number of houses')
    fig.set_xlabel(var)
    
    plt.show()
    
"""
The majority of the continuous variables contain outliers. Also most of the
variables are not normally distributed. We will have to fix this
in order to improve the performance of linear regression.

We can use discretization to fix this. Decision trees can be used to find the
right buckets.
"""

#%%

"""
Let's identify the outlier or rare values in these discrete
values. We will process them like they are categorical variables.
"""

for var in discrete:
    (data.groupby(var)[var].count() / np.float(len(data))).plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    plt.show()
    print()
    
#%%
    
no_labels_ls = []
for var in categorical:
    no_labels_ls.append(len(data[var].unique()))
    
tmp = pd.Series(no_labels_ls)
tmp.index = pd.Series(categorical)
tmp.plot.bar(figsize=(12,8))
plt.title('Number of categories in categorical variables')
plt.xlabel('Categorical variables')
plt.ylabel('Number of different categories')

"""
Most of the varaibles contain only a few labels. We will
have to deal with the high cardinality of the few that do.
"""

