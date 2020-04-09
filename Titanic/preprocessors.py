# -*- coding: utf-8 -*-
"""
Preprocessors.py
"""

# custom package for reading the data.
from data_management import load_dataset
from features import get_title, find_categorical_and_numerical_variables, impute_na, impute_age, top_code, rare_imputation, encode_categorical_variables

# for text / string processing
import re

# Plotting libraries
import matplotlib.pyplot as plt

# to divide into train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to handle data sets
import pandas as pd
import numpy as np

# to build models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, LeaveOneOut
import xgboost as xgb

# to evaluate the models
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


#%%

# load training dataset
data = load_dataset("train.csv")
data.head()

#%%

# Load submission dataset
submission = load_dataset("test.csv")
submission.head()

#%%

# extracts the number from the string
data['Cabin_numerical'] = data.Cabin.str.extract('(\d+)')
submission['Cabin_numerical'] = submission.Cabin.str.extract('(\d+)')

# converts the above variable to a float
data['Cabin_numerical'] = data['Cabin_numerical'].astype('float')
submission['Cabin_numerical'] = submission['Cabin_numerical'].astype('float')

# Captures the first letter of the string
data['Cabin_categorical'] = data['Cabin'].str[0]
submission['Cabin_categorical'] = submission['Cabin'].str[0]

# Count the number of cabins.
data_temp = data.Cabin.str.split()
submission_temp = submission.Cabin.str.split()

data_temp[ ~data_temp.isnull() ] = data_temp[ ~data_temp.isnull() ].str.len()
submission_temp[ ~submission_temp.isnull() ] = submission_temp[ ~submission_temp.isnull() ].str.len()

data['Number_of_cabins'] = data_temp
submission['Number_of_cabins'] = submission_temp

#%%

# Check out the first variables
data[['Cabin', 'Cabin_numerical', 'Cabin_categorical', 'Number_of_cabins']].head()

#%%

# Remove the now unecessary Cabin Variable
data.drop( labels='Cabin', inplace=True, axis=1 )
submission.drop( labels='Cabin', inplace = True, axis=1 )

#%%

# Extract the last part of the ticket as a number
data['Ticket_numerical'] = data.Ticket.apply(lambda s: s.split()[-1] )
data['Ticket_numerical'] = np.where(data.Ticket_numerical.str.isdigit(), data.Ticket_numerical, np.nan )
data['Ticket_numerical'] = data['Ticket_numerical'].astype('float')

submission['Ticket_numerical'] = submission.Ticket.apply(lambda s: s.split()[-1] )
submission['Ticket_numerical'] = np.where(submission.Ticket_numerical.str.isdigit(), submission.Ticket_numerical, np.nan )
submission['Ticket_numerical'] = submission['Ticket_numerical'].astype('float')

# Extract the first part of the ticket as a string
data['Ticket_categorical'] = data.Ticket.apply(lambda s: s.split()[0] )
data['Ticket_categorical'] = np.where( data.Ticket_categorical.str.isdigit(), np.nan, data.Ticket_categorical )

submission['Ticket_categorical'] = submission.Ticket.apply(lambda s: s.split()[0] )
submission['Ticket_categorical'] = np.where( submission.Ticket_categorical.str.isdigit(), np.nan, submission.Ticket_categorical )


data[['Ticket', 'Ticket_numerical', 'Ticket_categorical']].head()

#%%

data.Ticket_categorical.unique()

# There are still a lot of categories for Ticket_categorical.
# We should attempt to reduce it further.

#%%

# remove the non letter characters from the strings
data['Ticket_categorical'] = data.Ticket_categorical.apply( lambda x: re.sub("[^a-zA-Z^]", '', str(x)) )
data['Ticket_categorical'].str.upper()

submission['Ticket_categorical'] = submission.Ticket_categorical.apply( lambda x: re.sub("[^a-zA-Z]", '', str(x)) )
submission['Ticket_categorical'] = submission['Ticket_categorical'].str.upper()
data.Ticket_categorical.unique()

#%%

# drop the original variable
data.drop(labels='Ticket', inplace=True, axis=1)
submission.drop(labels='Ticket', inplace=True, axis=1)

#%%

data['Title'] = pd.Series([i.split(',')[1].split('.')[0].strip() for i in data['Name']])
submission['Title'] = pd.Series([i.split(',')[1].split('.')[0].strip() for i in submission['Name']])
data.Title.unique()

#%%
    
data['Title'] = data['Title'].apply(get_title)
submission['Title'] = submission['Title'].apply(get_title)

data['Title'].unique()

#%%

data.drop(labels='Name', inplace = True, axis = 1)
submission.drop(labels='Name', inplace = True, axis = 1)

#%%

# create a variable indicating family size (including the passenger)
# sums siblings and parents

data['Family_size'] = data['SibSp']+data['Parch']+1
submission['Family_size'] = submission['SibSp']+submission['Parch']+1

print(data.Family_size.value_counts()/ np.float(len(data)))

(data.Family_size.value_counts() / np.float(len(data))).plot.bar()

#%%

# Create a new variable to indicate if
# the passenger is a mother or not
# variable indicating if passenger was a mother
data['is_mother'] = np.where((data.Sex =='female')&(data.Parch>=1)&(data.Age>18),1,0)
submission['is_mother'] = np.where((submission.Sex =='female')&(submission.Parch>=1)&(submission.Age>18),1,0)

data.loc[data.is_mother == 1, ['Sex', 'Parch', 'Age', 'is_mother']].head()

#%%

print('There were {} mothers on the Titanic'.format(data.is_mother.sum()))

#%%

# Now let's look for missing data, outliers, cardinality, and rare labels
# in the newly created variables

data[['Cabin_numerical', 'Ticket_numerical', 'is_mother', 'Family_size']].isnull().mean()

#%%

# First we plot the distribution to find out
# if they are Gaussian or skewed. If Gaussian
# we will use the normal assumption otherwise we
# will use the interquartile range to find outliers

plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.Cabin_numerical.hist(bins=50)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Cabin number')

plt.subplot(1, 2, 2)
fig = data.Ticket_numerical.hist(bins=50)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Ticket number')

#%%

# let's visualise outliers with the boxplot and whiskers
plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = data.boxplot(column='Cabin_numerical')
fig.set_title('')
fig.set_ylabel('Cabin number')

plt.subplot(1, 2, 2)
fig = data.boxplot(column='Ticket_numerical')
fig.set_title('')
fig.set_ylabel('Ticket number')

# Cabin number doesn't have outliers
# Ticket number has outliers

#%%

# Ticket numerical
IQR = data.Ticket_numerical.quantile(0.75) - data.Ticket_numerical.quantile(0.25)
Lower_fence = data.Ticket_numerical.quantile(0.25) - (IQR * 3)
Upper_fence = data.Ticket_numerical.quantile(0.75) + (IQR * 3)
print('Ticket number outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
passengers = len(data[data.Ticket_numerical>Upper_fence]) / np.float(len(data))
print('Number of passengers with ticket values higher than {upperboundary}: {passengers}'.format(upperboundary=Upper_fence, \
                                                                                                 passengers=passengers))
#%%

# Time to check the number of missing values in
# each of our new variables

data[['Cabin_categorical', 'Ticket_categorical', 'Title']].isnull().mean()

# Cabin_categorical contains a lot of missing values

#%%

for var in ['Cabin_categorical', 'Ticket_categorical', 'Title']:
    print(var, ' contains ', len(data[var].unique()), ' labels')

#%%

# rare / unfrequent labels (less than 1% of passengers)
for var in ['Cabin_categorical', 'Ticket_categorical', 'Title']:
    print(100*data[var].value_counts() / np.float(len(data)))
    print()
    
# Cabin_categorical contains rare variables 'G' and 'T', unfortunately, even if we
# replace these variables, with rare it wouldn't help since the rare
# variable would also be rare and lead to overfitting. We will instead
# absorb these variables into the most frequent label.
    
# Ticket_categorical contains numerous rare variables, which we will replace with 'rare'

# Title doesn't contain any rare variables

#%%

# Let's group together the variables into categorical and numerical
# considering the newly created variables

categorical, numerical = find_categorical_and_numerical_variables(data)

#%%

numerical.remove("PassengerId")
categorical.remove("Survived")
    
#%%

# Time to deal with missing data

# print variables with missing data
for col in numerical:
    if data[col].isnull().mean() > 0:
        print( col, data[col].isnull().mean() )
        
# Since 'Age' and 'Ticket' contain less than 50% NA, we will create an
# additional variable with random sample imputation
        
# Since Cabin_numerical contains > 50% NA, we will impute NA by a far value
# in the distribution.

#%%

# Let's separate into train and test set

X_train, X_test, y_train, y_test = train_test_split(data, data.Survived, test_size=0.2,
                                                    random_state=95064)
X_train.shape, X_test.shape

#%%
    
# Age and Ticket_numerical
# add variable indicating missingness
    
for df in [X_train, X_test, submission]:
    for var in ['Age', 'Ticket_numerical']:
        df[var+ '_NA'] = np.where( df[var].isnull(), 1,0)
        
# replace by random sampling
for df in [X_train, X_test, submission]:
    df['Ticket_numerical'] = impute_na(X_train, df, 'Ticket_numerical')
    df['Age'] = impute_age(X_train, df)
    
# Cabin_numerical
extreme = X_train.Cabin_numerical.mean() + X_train.Cabin_numerical.std()*3
for df in [X_train, X_test, submission]:
    df.Cabin_numerical.fillna(extreme, inplace = True)
    
#%%
    
# Print categorical variables with missing data
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())
        
# We can impute Embarked NA by the most frequent category because NA is low
# Cabin_categorical will be imputed by 'Missing', because NA is high.
        
#%%
        
for df in [X_train, X_test, submission]:
    df['Embarked'].fillna( X_train['Embarked'].mode()[0], inplace = True)
    df['Cabin_categorical'].fillna('Missing', inplace = True )
    df['Number_of_cabins'].fillna(0, inplace = True)
    
#%%
    
# Time to double check for the abscence of null values
X_train.isnull().sum()

#%%

X_test.isnull().sum()

#%%

submission.isnull().sum()

#%%

# Fare in the submission dataset contains one null value, I will replace by the median 
submission.Fare.fillna(X_train.Fare.median(), inplace=True)

#%%



for df in [X_train, X_test, submission]:
    df['Age'] = top_code(df, 'Age', 73)
    df['SibSp'] = top_code(df, 'SibSp', 4)
    df['Parch'] = top_code(df, 'Parch', 2)
    df['Family_size'] = top_code(df, 'Family_size', 7)
    
#%%
    
# Let's verify that it worked.
for var in ['Age', 'SibSp', 'Parch', 'Family_size']:
    print(var, ' max value: ', X_train[var].max() )
    
#%%
    
# find quantiles and discretise train set
X_train['Fare'], bins = pd.qcut(x=X_train['Fare'], q=8, retbins=True, precision=3, duplicates='raise')
X_test['Fare'] = pd.cut(x = X_test['Fare'], bins=bins, include_lowest=True)
submission['Fare'] = pd.cut(x = submission['Fare'], bins=bins, include_lowest=True)

#%%

submission.Fare.isnull().sum()

#%%

t1 = X_train.groupby(['Fare'])['Fare'].count() / np.float(len(X_train))
t2 = X_test.groupby(['Fare'])['Fare'].count() / np.float(len(X_test))
t3 = submission.groupby(['Fare'])['Fare'].count() / np.float(len(submission))

temp = pd.concat([t1,t2,t3], axis=1)
temp.columns = ['train', 'test', 'submission']
temp.plot.bar(figsize=(12,6))

#%%

# find quantiles and discretise train set
X_train['Ticket_numerical'], bins = pd.qcut(x=X_train['Ticket_numerical'], q=8, retbins=True, precision=3, duplicates='raise')
X_test['Ticket_numerical'] = pd.cut(x = X_test['Ticket_numerical'], bins=bins, include_lowest=True)
submission['Ticket_numerical_temp'] = pd.cut(x = submission['Ticket_numerical'], bins=bins, include_lowest=True)


#%%

X_test.Ticket_numerical.isnull().sum()

#%%

submission.Ticket_numerical_temp.isnull().sum()

#%%

submission[submission.Ticket_numerical_temp.isnull()][['Ticket_numerical', 'Ticket_numerical_temp']]

#%%

X_train.Ticket_numerical.unique()

#%%

X_train.Ticket_numerical.unique()[0]

#%%

submission.loc[submission.Ticket_numerical_temp.isnull(), 'Ticket_numerical_temp'] = X_train.Ticket_numerical.unique()[0]
submission.Ticket_numerical_temp.isnull().sum()

#%%

submission['Ticket_numerical'] = submission['Ticket_numerical_temp']
submission.drop( labels = ['Ticket_numerical_temp'], inplace = True, axis=1 )
submission.head()

#%%

for var in categorical:
    print( var, X_train[var].value_counts()/np.float(len(X_train)))
    print()
        
#%%    
        
rare_imputation( X_train, X_test, submission, 'Cabin_categorical', 'frequent')
rare_imputation( X_train, X_test, submission, 'Ticket_categorical', 'rare')     
        
#%%

# let's verify that it worked
for var in categorical:
    print( var, X_train[var].value_counts()/np.float(len(X_train)))
    print()


#%%
    
for var in categorical:
    print( var, submission[var].value_counts()/np.float(len(submission)))
    print()
    
#%%
    
categorical
    
#%%

for df in [X_train, X_test, submission]:
    df['Sex'] = pd.get_dummies(df.Sex, drop_first=True)

#%%
    
X_train.Sex.unique()

#%%

X_test.Sex.unique()

#%%

submission.Sex.unique()

#%%
    
# eoncode labels in categorical vars
for var in categorical:
    encode_categorical_variables( X_train, X_test, submission, var, 'Survived')

#%%
    
# parse discretized variables to object before encoding
for df in [X_train, X_test, submission]:
    df.Fare = df.Fare.astype('O')
    df.Ticket_numerical = df.Ticket_numerical.astype('O')
    
#%%
    
# encode labels
for var in ['Fare', 'Ticket_numerical']:
    print(var)
    encode_categorical_variables( X_train, X_test, submission, var,'Survived')
    
#%%
    
X_train.head()

#%%

X_train.describe()

#%%

variables_that_need_scaling = ['Pclass', 'Age', 'Sibsp', 'Parch', 'Cabin_number', 'Family_size']

#%%

training_vars = [ var for var in X_train.columns if var not in ['PassengerId', 'Survived']]

#%%

# fit scaler
scaler = MinMaxScaler() # create an instance
scaler.fit(X_train[training_vars]) # fit the scaler to the train set and then transform it

#%%

# Cross validate model with Kfold stratified cross val
#kfold = StratifiedKFold(n_splits=10)
loo = LeaveOneOut()
loo.get_n_splits(X_train)
 
#%%

xgb_model = xgb.XGBClassifier()

eval_set = [(X_test[training_vars], y_test)]
xgb_model.fit(X_train[training_vars], y_train, eval_metric="auc", eval_set=eval_set, verbose=False)

pred = xgb_model.predict_proba(X_train[training_vars])
print('xgb train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = xgb_model.predict_proba(X_test[training_vars])
print('xgb test roc-auc: {}'.format( roc_auc_score( y_test, pred[:,1])))


#%%

# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
#rf_param_grid = {"max_depth": [None],
#              "max_features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#              "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
#              "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#              "bootstrap": [False, True],
#              "n_estimators" :[100, 150, 200, 250, 300],
#              "criterion": ["gini"]}

rf_param_grid = {'n_estimators':[10, 20, 40, 80, 100, 200, 400, 800],
              'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
              'min_samples_split':[2,3,4,5,6,7,8],
              'min_samples_leaf':[1,2,3,4,5],
              'bootstrap':[True,False] }


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=loo, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train[training_vars],y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_
gsRFC.best_params_

#%%

pred = RFC_best.predict_proba(X_train[training_vars])
print('RF train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = RFC_best.predict_proba(X_test[training_vars])
print('RF test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

#%%

# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2,3,4,5,6,7,8,9,10],
              "learning_rate":  [ 0.28, 0.29 , 0.3, 0.31, 0.32 ]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train[training_vars],y_train)

ada_best = gsadaDTC.best_estimator_

# Best score
gsadaDTC.best_score_
gsadaDTC.best_params_

#%%

pred = ada_best.predict_proba(X_train[training_vars])
print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = ada_best.predict_proba(X_test[training_vars])
print('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

#%%

logit_model = LogisticRegression()

logit_param_grid = {"C": np.linspace(1,1001,1000)}

gslogit = GridSearchCV( logit_model, param_grid = logit_param_grid, cv=kfold, scoring = "accuracy", n_jobs = 4, verbose = 1)

gslogit.fit(X_train[training_vars], y_train)

logit_best = gslogit.best_estimator_

# Best Score
gslogit.best_score_
gslogit.best_params_

#%%

pred = logit_best.predict_proba(scaler.transform(X_train[training_vars]))
print('Logit train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = logit_best.predict_proba(scaler.transform(X_test[training_vars]))
print('Logit test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

#%%

pred_ls = []
#for model in [xgb_model, RFC_best, ada_best, logit_best]:
for model in [xgb_model]:
    pred_ls.append(pd.Series(model.predict_proba(X_test[training_vars])[:,1]))

final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)
print('Ensemble test roc-auc: {}'.format(roc_auc_score(y_test,final_pred)))

#%%

tpr, tpr, thresholds = metrics.roc_curve(y_test, final_pred)
thresholds

#%%

accuracy_ls = []
for thres in thresholds:
    y_pred = np.where( final_pred>thres, 1, 0 )
    accuracy_ls.append( metrics.accuracy_score( y_test, y_pred, normalize=True))
    
accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],
                         axis=1)
accuracy_ls.columns = ['thresholds', 'accuracy']
accuracy_ls.sort_values( by='accuracy', ascending=False, inplace=True)
accuracy_ls.head()

#%%

## Submission to Kaggle ##

pred_ls = []
#for model in [xgb_model, RFC_best, ada_best, logit_best]:
for model in [xgb_model]:
    pred_ls.append(pd.Series(model.predict_proba(submission[training_vars])[:,1]))
    
final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)

#%%

final_pred = pd.Series( np.where(final_pred>0.5, 1, 0))

#%%

temp = pd.concat([submission.PassengerId, final_pred], axis=1)
temp.columns = ['PassengerId', 'Survived']


#%%

test_Survived = pd.Series(votingC.predict(test), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)

#%%

temp.to_csv('submit_titanic.csv', index=False)

#%%

importance = pd.Series(RFC_best.feature_importances_)
importance.index = training_vars
importance.sort_values(inplace = True, ascending=False)
importance.plot.bar(figsize=(12,6))

#%%

importance = pd.Series(xgb_model.feature_importances_)
importance.index = training_vars
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(12,6))