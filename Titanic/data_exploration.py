# -*- coding: utf-8 -*-
"""
data_exploration.py
"""

# custom package for reading the data.
import data_management as dm
from matplotlib import pyplot as plt

# to handle data sets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

#%%

# load training dataset
data = dm.load_dataset("train.csv")
data.head()

#%%

# Load submission dataset
submission = dm.load_dataset("test.csv")
submission.head()

#%%

data.info()

#%%

data.shape

#%%

data.describe()

#%%

# Inspecting data types
data.dtypes

#%%

print('Number of PassengerId labels: ', len(data.PassengerId.unique()) )
print('Number of Passengers on the titanic: ', len(data.Ticket.unique()) )
# The number of labels is equal to the number of
# passengers. Thus this PassengerId isn't helpful.

#%%

categorical = [ var for var in data.columns if data[var].dtype == 'O' ]
print('There are {} categorical variables'.format(len(categorical)) )


#%%

numerical = [var for var in data.columns if data[var].dtype != 'O' ]
print('There are {} numerical variables'.format(len(numerical)) )

#%%

#Let's take a look at the categorical variables
data[categorical].head()

#Note that Cabin and Ticket contain both letters
#and numbers. We can extract the numerical part
#and the non-numerical part to generate 2 new
#for possible additional value.

#%%

#Let's take a look at the numerical variables
data[numerical].head()

#-Discrete Variables: Pclass, SibSp, and Parch
#-Continuous Variables: Fare and Age
#-binary Variable - Survived (also the target variable)
#-Id Variable - useless

#%%


# Let's look at the percentage of missing data in variables
data.isnull().mean()*100

#Age is missing 20%
#Cabin is missing 77%
#Embarked is missing <1%

#%%

numerical = [var for var in numerical if var not in ['Survived', 'PassengerId'] ]
numerical

#%%

"""
Let's make boxplots to visualize the outliers
in the continuous variables (Age and Fare)
"""
fig, axes = plt.subplots(2,1, figsize=(15,6))

sns.boxplot(data.Age, ax=axes[0])
axes[0].set_title('Age Distribution',fontsize=18)
axes[0].set_xlabel('Age',fontsize=14)
sns.boxplot(data.Fare, ax=axes[1])
axes[1].set_title('Fare Distribution',fontsize=18)
axes[1].set_xlabel('Fare',fontsize=14)
plt.tight_layout( h_pad=2.0)
#plt.grid(True, alpha=0.6)
#plt.title("Box Plots", fontsize=18)
#plt.xlabel("Values  ->", fontsize=14)
#plt.ylabel("Features", fontsize=14)


#%%

"""
Now we plot the distributions to check
if our variables are Gaussian or skewed.
If normal we'll use the normal assumption. Otherwise
we'll use the interquantile range to find the outliers.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,6))
plt.tight_layout
ax1.hist(data.Age, bins=10,
              density=True, ec='white', alpha = 0.7)
ax1.set_title('Age Distribution',fontsize=18)
ax1.set_xlabel('Age', fontsize=14)

ax2.hist(data.Fare, bins=10,
              density=True, ec='white', alpha = 0.7)
ax2.set_title('Fare Distribution',fontsize=18)
ax2.set_xlabel('Fare', fontsize=14)

#plt.subplot(1,2,2)
#fig = plt.hist(data.Fare, bins=10, color='lightblue',
#               density=True, ec='white')
#fig.set_ylabel('Number of passengers')
#fig.set_xlabel('Fare')

#%%

# Let's change Fare to the average ticket price per person.

temp = data.groupby('Ticket')['Ticket'].count()
for i in range(len(data)):
    data.Fare[i] = data.Fare[i]/(temp[data.Ticket[i]])

#%%
    
plt.figure( figsize=(15,6) )
plt.subplot(1,2,1)
fig = data.boxplot(column='Fare')
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Age')

plt.subplot(1,2,2)
fig = data.Fare.hist(bins=20)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Fare')

#%%
# Let's identify the outliers

# Age

Upper_boundary = data.Age.mean() + 3*data.Age.std()
Lower_boundary = data.Age.mean() - 3*data.Age.std()
print('Fare outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary = Lower_boundary, upperboundary = Upper_boundary) )

# Fare
IQR = data.Fare.quantile(0.75) - data.Fare.quantile(0.25)
lower_fence = data.Fare.quantile(0.25) - 3*IQR
upper_fence = data.Fare.quantile(0.75) + 3*IQR
print('Fare outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=lower_fence, upperboundary=upper_fence) )


#%%
sex = round(100*(data['Sex'].value_counts()/len(data)),1)

plt.bar(['Female', 'Male'], sex[['female','male']],
        alpha=0.7, width=0.6)

plt.grid(True, alpha=0.3)
#plt.xlabel("Target  ->", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.title("Distribution of Sex", fontsize=18)

# Remove top and right spines:
ax = plt.gca() # Get current axis (gca)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.annotate(str(sex['female']), xy=(0, sex['female']), 
             xytext=(0,sex['female']+3), ha = 'center',
             bbox={'boxstyle': 'round', 'pad': 0.5, 'facecolor': 
                   'orange', 'edgecolor': 'orange', 'alpha': 0.6},
             arrowprops={'arrowstyle':"wedge,tail_width=0.5", 
                         'alpha':0.6, 'color': 'orange'})
plt.annotate(str(sex['male']), xy=(1, sex['male']), 
             xytext=(1, sex['male']+3), ha = 'center',
             bbox={'boxstyle': 'round', 'pad': 0.5, 'facecolor': 
                   'orange', 'edgecolor': 'orange', 'alpha': 0.6},
             arrowprops={'arrowstyle':"wedge,tail_width=0.5", 
                         'alpha':0.6, 'color': 'orange'})
plt.ylim([0, 80])
#plt.figure( figsize=(15,6) )
#plt.subplot(1,2,1)
#fig = sex[['male','female']].plot.bar()
#fig.set_ylabel('Percentage')
#fig.set_xlabel('Sex')
#fig.set_title('Distribution')

#plt.subplot(1,2,2)
#fig = data.groupby('Sex')['Survived'].mean()[['male','female']].plot.bar()
#fig.set_ylabel('Percent')
#fig.set_xlabel('Sex')
#fig.set_title('Probability of Survival')
#fig.set_ylim((0,0.8))

#%%
Sibsp = round(100*(data['SibSp'].value_counts()/len(data)),1)
cat = list(np.sort(Sibsp.index))
height = Sibsp[list(np.sort(Sibsp.index))]

plt.bar( cat, height,
        alpha=0.7, width=0.6)

plt.grid(True, alpha=0.0)
#plt.xlabel("Target  ->", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.title("Distribution of SibSp", fontsize=18)

# Remove top and right spines:
ax = plt.gca() # Get current axis (gca)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim([0, 84])

for (i,j) in zip(cat, height):
    plt.annotate(str(j), xy=(i, j), 
             xytext=(i,j+0.1*max(height)), ha = 'center',
             bbox={'boxstyle': 'round', 'pad': 0.5, 'facecolor': 
                   'orange', 'edgecolor': 'orange', 'alpha': 0.6},
             arrowprops={'arrowstyle':"wedge,tail_width=0.5", 
                         'alpha':0.6, 'color': 'orange'})


#%%
Parch = round(100*(data['Parch'].value_counts()/len(data)),1)
cat = list(np.sort(Parch.index))
height = Parch[list(np.sort(Parch.index))]

plt.bar( cat, height,
        alpha=0.7, width=0.6)

plt.grid(True, alpha=0.0)
#plt.xlabel("Target  ->", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.title("Distribution of Parch", fontsize=18)

# Remove top and right spines:
ax = plt.gca() # Get current axis (gca)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim([0, max(height) + 0.2*max(height)])

for (i,j) in zip(cat, height):
    plt.annotate(str(j), xy=(i, j), 
             xytext=(i,j+0.1*max(height)), ha = 'center',
             bbox={'boxstyle': 'round', 'pad': 0.5, 'facecolor': 
                   'orange', 'edgecolor': 'orange', 'alpha': 0.6},
             arrowprops={'arrowstyle':"wedge,tail_width=0.5", 
                         'alpha':0.6, 'color': 'orange'})

#plt.figure( figsize=(15,6) )
#plt.subplot(1,2,1)
#fig = Parch[[0,1,2,3,4,5,6]].plot.bar()
#fig.set_ylabel('Percentage')
#fig.set_xlabel('Parch')
#fig.set_title('Distribution')

#plt.subplot(1,2,2)
#fig = data.groupby('Sex')['Survived'].mean()[['male','female']].plot.bar()
#fig.set_ylabel('Percent')
#fig.set_xlabel('Sex')
#fig.set_title('Probability of Survival')
#fig.set_ylim((0,0.8))

#%%
Pclass = round(data['Pclass'].value_counts()/len(data)*100,1)
cat = sorted(list(Pclass.index))
height = Pclass[cat]

plt.bar( cat, height,
        alpha=0.7, width=0.6)

plt.grid(True, alpha=0.0)
#plt.xlabel("Target  ->", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.title("Distribution of Pclass", fontsize=18)

# Remove top and right spines:
ax = plt.gca() # Get current axis (gca)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim([0, max(height) + 0.2*max(height)])

for (i,j) in zip(cat, height):
    plt.annotate(str(j), xy=(i, j), 
             xytext=(i,j+0.1*max(height)), ha = 'center',
             bbox={'boxstyle': 'round', 'pad': 0.5, 'facecolor': 
                   'orange', 'edgecolor': 'orange', 'alpha': 0.6},
             arrowprops={'arrowstyle':"wedge,tail_width=0.5", 
                         'alpha':0.6, 'color': 'orange'})

#plt.figure( figsize=(15,6) )
#plt.subplot(1,2,1)
#fig = Pclass[[1,2,3]].plot.bar()
#fig.set_ylabel('Percentage')
#fig.set_xlabel('Pclass')
#fig.set_title('Distribution')

#plt.subplot(1,2,2)
#fig = data.groupby('Sex')['Survived'].mean()[['male','female']].plot.bar()
#fig.set_ylabel('Percent')
#fig.set_xlabel('Sex')
#fig.set_title('Probability of Survival')
#fig.set_ylim((0,0.8))

#%%
Embarked = round(100*(data['Embarked'].value_counts()/len(data)),1)
cat = list(Embarked.index)
cat.sort()
height = Embarked[cat]

plt.bar( cat, height,
        alpha=0.7, width=0.6)

plt.grid(True, alpha=0.0)
#plt.xlabel("Target  ->", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.title("Distribution of Pclass", fontsize=18)

# Remove top and right spines:
ax = plt.gca() # Get current axis (gca)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim([0, max(height) + 0.2*max(height)])

for (i,j) in zip(cat, height):
    plt.annotate(str(j), xy=(i, j), 
             xytext=(i,j+0.1*max(height)), ha = 'center',
             bbox={'boxstyle': 'round', 'pad': 0.5, 'facecolor': 
                   'orange', 'edgecolor': 'orange', 'alpha': 0.6},
             arrowprops={'arrowstyle':"wedge,tail_width=0.5", 
                         'alpha':0.6, 'color': 'orange'})


#plt.figure( figsize=(15,6) )
#plt.subplot(1,2,1)
#fig = Embarked[['C','Q','S']].plot.pie()
#fig.set_ylabel('Percentage')
#fig.set_xlabel('Embarked')
#fig.set_title('Distribution')

#plt.subplot(1,2,2)
#fig = data.groupby('Sex')['Survived'].mean()[['male','female']].plot.bar()
#fig.set_ylabel('Percent')
#fig.set_xlabel('Sex')
#fig.set_title('Probability of Survival')
#fig.set_ylim((0,0.8))

#%%
plt.figure( figsize=(6,15) )

plt.subplot(1,1,1)
fig = sex[['male','female']].plot.pie()
plt.subplot(1,2,1)
fig = Sibsp[[0,1,2,3,4,5,8]].plot.pie()
plt.subplot(2,1,1)
fig = Parch[[0,1,2,3,4,5,6]].plot.pie()
plt.subplot(2,2,1)
fig = Pclass[[1,2,3]].plot.pie()
plt.subplot(3,2,1)
fig = Embarked[['C','Q','S']].plot.pie()

plt.show()

#%%

ax = data.groupby('SibSp')['Survived'].mean().plot.bar(x='SibSp', y='Survived' )
ax.set_ylabel('Probability of Survival')



#%%

ax = data.groupby('Parch')['Survived'].mean().plot.bar(x='Parch', y='Survived' )
ax.set_ylabel('Probability of Survival')

#%%

plt.figure( figsize=(15,6) )
plt.subplot(1,2,1)
fig = data[data['Survived']==0].Age.hist(bins=20)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Age')
fig.set_title('Survived = 0')
fig.set_xlim((0,80))
fig.set_ylim((0,60))

plt.subplot(1,2,2)
fig = data[data['Survived']==1].Age.hist(bins=20)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Age')
fig.set_title('Survived = 1')
fig.set_xlim((0,80))
fig.set_ylim((0,60))

#%%

plt.figure( figsize=(15,6) )
plt.subplot(1,2,1)
fig = data[data['Survived']==0].Age.plot.density()
fig.set_ylabel('Density')
fig.set_xlabel('Age')
fig.set_title('Survived = 0')
fig.set_xlim((0,80))
fig.set_ylim((0,0.035))

plt.subplot(1,2,2)
fig = data[data['Survived']==1].Age.plot.density()
fig.set_ylabel('Density')
fig.set_xlabel('Age')
fig.set_title('Survived = 1')
fig.set_xlim((0,80))
fig.set_ylim((0,0.035))


#%%

data.groupby(['Pclass','Sex'])['Survived'].mean()

#%%

data.groupby('Embarked')['Survived'].mean()

#%%

data.groupby(['Embarked','Pclass'])['Survived'].mean()

#%%

data.groupby(['Embarked','Sex'])['Survived'].mean()

#%%

# Explore SibSp feature vs Survived
g = sns.factorplot(x="SibSp",y="Survived",data=data, kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

#%%
# Search for rare values in the discrete variables
for var in ['Pclass', 'SibSp', 'Parch']:
    print( data[var].value_counts() / np.float(len(data)) )
    print()

# Note that:
# - Pclass has no rare variables.
# - SibSp the number of siblings/spouses greater than 4 are rare. Thus we can cap it
#   at 4 (Top-coding)
# - Parch the number of parents/childresn greater than 2 are rare. Thus we will cap it
#   at 2 (Top-coding)

#%%
    
for var in categorical:
    print( var, ' contains ', len(data[var].unique()), 'labels')
    