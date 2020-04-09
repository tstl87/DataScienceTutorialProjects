# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:24:30 2019

@author: striguei
"""

# to handle datasets
import pandas as pd
import numpy as np

import data_management as dm
import data_exploration as de

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

X_train, X_test, y_train, y_test = train_test_split( data, data.SalePrice,
                                                    test_size = 0.1, random_state = 0 )

X_train.shape, X_test.shape

#%%

def elapsed_years(df, var):
    # capture difference between year variable and the year the house was sold
    df[var] = df['YrSold'] - df[var]
    return df

#%%
    
for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    X_train = elapsed_years(X_train, var)
    X_test = elapsed_years(X_test, var)
    submission = elapsed_years(submission, var)
    
#%%
    
X_train[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()

#%%

# drop YrSold
X_train.drop('YrSold', axis=1, inplace = True)
X_test.drop('YrSold', axis=1, inplace = True)
submission.drop('YrSold', axis=1, inplace = True)

#%%

"""
print variables with missing data. Since we created
new time variables, then we will now treat tham as
continuous numerical variables.
"""

# remove YrSold since it is out of the dataset
de.year_vars.remove('YrSold')

# Examine the percentage of missing values
for col in de.numerical + de.year_vars:
    if X_train[col].isnull().mean() > 0:
        print( col, X_train[col].isnull().mean())
        
#%%

# Time to add variable indicating missing input and impute median value
for df in [X_train, X_test, submission]:
    for var in ['LotFrontage', 'GarageYrBlt']:
        df[var+'_NA'] = np.where(df[var].isnull(),1,0)
        df[var].fillna(X_train[var].median(), inplace = True)
        
for df in [X_train, X_test, submission]:
    df['MasVnrArea'].fillna(X_train.MasVnrArea.median(), inplace = True)

#%%
    
for col in de.discrete:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())
        
#%%
        
for col in de.categorical:
    if X_train[col].isnull().mean()>0:
        print( col, X_train[col].isnull().mean())
    
        
#%%
    
# replace each null value with a 'Missing' label
for df in [X_train, X_test, submission]:
    for var in de.categorical:
        df[var].fillna('Missing', inplace=True)
        
#%%

# verify that all null values are gone.
for var in X_train.columns:
    if X_train[var].isnull().sum()>0:
        print( var, X_train[var].isnull().sum())
        
#%%
        
# verify that all null values are gone.
for var in X_train.columns:
    if X_test[var].isnull().sum()>0:
        print( var, X_test[var].isnull().sum())
        
#%%
        
submission_vars = []
# verify that all null values are gone.
for var in X_train.columns:
    if var != 'SalePrice' and submission[var].isnull().sum()>0 :
        print( var, submission[var].isnull().sum())
        submission_vars.append(var)
        
#%%
        
"""
Fill null values with median value for variables with null value
in the submission.
"""

for var in submission_vars:
    submission[var].fillna(X_train[var].median(), inplace=True)
    
#%%
    
def tree_binariser(var):
    score_ls = [] # list for storing the mean squared error
    
    for tree_depth in [1,2,3,4]:
        
        tree_model = DecisionTreeRegressor(max_depth=tree_depth)
        
        # train the model using 3 fold cross validation
        scores = cross_val_score(tree_model, X_train[var].to_frame(), y_train, 
                                 cv=3, scoring = 'neg_mean_squared_error')
        score_ls.append(np.mean(scores))
        
    # find the depth with the smallest mean squared error
    depth = [1,2,3,4][np.argmin(score_ls)]
    
    # transform the variable using the tree
    tree_model = DecisionTreeRegressor(max_depth=depth)
    tree_model.fit(X_train[var].to_frame(), X_train.SalePrice)
    X_train[var] = tree_model.predict(X_train[var].to_frame())
    X_test[var] = tree_model.predict(X_test[var].to_frame())
    submission[var] = tree_model.predict(submission[var].to_frame())
    
#%%
    
for var in de.numerical:
    tree_binariser(var)
    
#%%
    
X_train[de.numerical].head()

#%%

"""
let's explore how many different buckets we have now among our
engineered continuous variables
"""

for var in de.numerical:
    print(var, len(X_train[var].unique()))
    
#%%
    
for var in de.numerical:
    X_train.groupby(var)['SalePrice'].mean().plot.bar()
    plt.title(var)
    plt.ylabel('Mean House Price')
    plt.xlabel('Discretized continuous variable')
    plt.show()
    
"""
Note that the mean house price increases with the value
of the bucket. Thus we now have a monotonic distribution
between the numerical variable and the target.
"""

#%%

def rare_imputation(variable):
    # find frequent labels / discrete numbers
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    frequent_cat = [x for x in temp.loc[temp>0.03].index.values]
    
    X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')
    X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
    submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], 'Rare')
    
#%%
    
# Convert the variables into ints like in the training set
for var in ['BsmtFullBath', 'BsmtHalfBath', 'GarageCars']:
    submission[var] = submission[var].astype('int')
    
#%%
    
# find the infrequent labels in categorical variables
# and replace by Rare
for var in de.categorical:
    rare_imputation(var)

# do the same discrete variables
for var in de.discrete:
    rare_imputation(var)
    
#%%
    
# check for missing values in submission dataset
for var in X_train.columns:
    if var != 'SalePrice' and submission[var].isnull().sum() > 0:
        print(var, submission[var].isnull().sum())
        submission_vars.append(var)
        
#%%
        
# Let's verify that it worked
for var in de.categorical:
    ( X_train.groupby(var)[var].count() / np.float(len(X_train))).plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    plt.show()
    
#%%
    
for var in de.discrete:
    (X_train.groupby(var)[var].count() / np.float(len(X_train))).plot.bar()
    plt.ylabel('Percentage of observations per label')
    plt.title(var)
    plt.show()
    
#%%%
    
"""
Encode categorical and discrete variables
"""

def encode_categorical_variables(var, target):
    # make label to price dictionary
    ordered_labels = X_train.groupby([var])[target].mean().to_dict()
    
    # encode variables
    X_train[var] = X_train[var].map(ordered_labels)
    X_test[var] = X_test[var].map(ordered_labels)
    submission[var] = submission[var].map(ordered_labels)
    
# encode labnels in categorical vars
for var in de.categorical:
    encode_categorical_variables(var, 'SalePrice')
    
# encode labels in discrete vars
for var in de.discrete:
    encode_categorical_variables(var, 'SalePrice')
    
#%%
    
# sanity check: let's look for NA values
for var in X_train.columns:
    if var != 'SalePrice' and submission[var].isnull().sum() > 0:
        print(var, submission[var].isnull().sum())
        
#%%
        
# time to revisit the data set.
X_train.head()

#%%

X_train.describe()

#%%

# let's create a list of the training variables
training_vars = [var for var in X_train.columns if var not in ['Id', 'SalePrice']]

print('total number of variables to use for training: ', len(training_vars))

#%%

training_vars

#%%

# fit scaler
scaler = MinMaxScaler() # create an instance
scaler.fit(X_train[training_vars]) # fit the scaler to the train set for later use

#%%

xgb_model = xgb.XGBRegressor()

eval_set = [(X_test[training_vars], np.log(y_test))]
xgb_model.fit(X_train[training_vars], np.log(y_train), eval_set=eval_set, verbose=False)

pred = xgb_model.predict(X_train[training_vars])
print('xgb train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))
print('xgb train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))
print()
pred = xgb_model.predict(X_test[training_vars])
print('xgb test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))
print('xgb test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
print()

#%%

rf_model = RandomForestRegressor(n_estimators=800, max_depth=6)
rf_model.fit(X_train[training_vars], np.log(y_train))

pred = rf_model.predict(X_train[training_vars])
print('rf train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))
print('rf train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))

print()
pred = rf_model.predict(X_test[training_vars])
print('rf test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))
print('rf test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
print()

#%%

SVR_model = SVR()
SVR_model.fit(scaler.transform(X_train[training_vars]), np.log(y_train))

pred = SVR_model.predict(X_train[training_vars])
print('SVR train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))
print('SVR train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))

print()
pred = SVR_model.predict(X_test[training_vars])
print('SVR test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))
print('SVR test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))
print()

#%%

lin_model = Lasso(random_state=2909, alpha=0.005)
lin_model.fit(scaler.transform(X_train[training_vars]), np.log(y_train))

pred = lin_model.predict(scaler.transform(X_train[training_vars]))
print('Lasso Linear Model train mse: {}'.format(mean_squared_error(y_train, np.exp(pred))))
print('Lasso Linear Model train rmse: {}'.format(sqrt(mean_squared_error(y_train, np.exp(pred)))))

print()
pred = lin_model.predict(scaler.transform(X_test[training_vars]))
print('Lasso Linear Model test mse: {}'.format(mean_squared_error(y_test, np.exp(pred))))
print('Lasso Linear Model test rmse: {}'.format(sqrt(mean_squared_error(y_test, np.exp(pred)))))


#%%

# make predictions for the submission dataset
final_pred = pred = lin_model.predict(scaler.transform(submission[training_vars]))

#%%

temp = pd.concat([submission.Id, pd.Series(np.exp(final_pred))], axis=1)
temp.columns = ['Id', 'SalePrice']
temp.head()

#%%

temp.to_csv('submit_housesale.csv', index=False)