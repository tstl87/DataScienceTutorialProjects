# -*- coding: utf-8 -*-
"""
features.py
"""

import re
import numpy as np

#%%

def impute_age(X_train, df):
    
    Age_mean = X_train.groupby(['Sex','Pclass'])['Age'].mean()
    temp = df.copy()
    
    temp.loc[ (temp.Sex == 'female') & (temp.Pclass == 1) & (temp.Age.isnull())] = Age_mean[0]
    temp.loc[ (temp.Sex == 'female') & (temp.Pclass == 2) & (temp.Age.isnull())] = Age_mean[1]
    temp.loc[ (temp.Sex == 'female') & (temp.Pclass == 3) & (temp.Age.isnull())] = Age_mean[2]
    temp.loc[ (temp.Sex == 'male') & (temp.Pclass == 1) & (temp.Age.isnull())] = Age_mean[3]
    temp.loc[ (temp.Sex == 'male') & (temp.Pclass == 2) & (temp.Age.isnull())] = Age_mean[4]
    temp.loc[ (temp.Sex == 'male') & (temp.Pclass == 3) & (temp.Age.isnull())] = Age_mean[5]
    return temp['Age']
    
#%%
    
def impute_na( X_train, df, variable):
    
    # make a temporary copy of df
    temp = df.copy()
    
    # extract random samples from the train set to fill the NA
    random_sample = X_train[variable].dropna().sample(temp[variable].isnull().sum(), 
        random_state=0)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = temp[ temp[variable].isnull() ].index
    temp.loc[ temp[variable].isnull(), variable] = random_sample
    
    return temp[variable]

#%%
    
# Time to topcode the data we commented on earlier
def top_code( df, variable, top ):
    return np.where( df[variable] > top, top, df[variable] )

#%%

def rare_imputation(X_train, X_test, submission, variable, which='rare'):    
    # find frequent labels
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    frequent_cat = [x for x in temp.loc[temp>0.01].index.values]
    
    # create new variables, with Rare labels imputed
    if which=='frequent':
        # find the most frequent category
        mode_label = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], mode_label)
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], mode_label)
        submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], mode_label)
    
    else:
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
        submission[variable] = np.where(submission[variable].isin(frequent_cat), submission[variable], 'Rare')
        
#%%
        
def encode_categorical_variables(X_train, X_test, submission, var, target):
    
    # make label to risk dictionary
    ordered_labels = X_train.groupby([var])[target].mean().to_dict()
    
    # encode variables
    X_train[var] = X_train[var].map(ordered_labels)
    X_test[var] = X_test[var].map(ordered_labels)
    submission[var] = submission[var].map(ordered_labels)
    
#%%
    
def get_title(title):
    
    line = title
    if (line in ['Mrs', 'Mme',]):
        return 'Mrs'
    elif (line in ['Mr']):
        return 'Mr'
    elif (line in ['Miss', 'Mlle',]):
        return 'Miss'
    elif (line in ['Master']):
        return 'Master'
    else:
        return 'Other'
    
#%%
        
def find_categorical_and_numerical_variables( data ):
    cat_vars = [ col for col in data.columns if( data[col].dtypes == 'O' or len(data[col].unique()) <= 10)]
    num_vars = [ col for col in data.columns if data[col].dtypes != 'O' and len(data[col].unique()) > 10]
        
    return cat_vars, num_vars