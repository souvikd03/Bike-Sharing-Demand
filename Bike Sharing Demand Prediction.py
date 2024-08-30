#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

# 1. Data Understanding and Exploration

# Import Libraries
import pandas as pd
import numpy as np

# Reading the dataset
dataset = pd.read_csv("BoomBikes.csv")

# Displaying the first few rows of the dataset
dataset.head()

# Displaying the shape of the dataset
dataset.shape

# Displaying the column names
dataset.columns

# Displaying the statistical summary
dataset.describe()

# Displaying the data types and null values
dataset.info()

# Assigning string values to different seasons instead of numeric values

# 1 = Spring
dataset.loc[(dataset['season'] == 1), 'season'] = 'Spring'

# 2 = Summer
dataset.loc[(dataset['season'] == 2), 'season'] = 'Summer'

# 3 = Fall
dataset.loc[(dataset['season'] == 3), 'season'] = 'Fall'

# 4 = Winter
dataset.loc[(dataset['season'] == 4), 'season'] = 'Winter'

# Displaying value counts of the 'Season' column
dataset['season'].astype('category').value_counts()

# Displaying value counts of the 'yr' column (0 = 2018, 1 = 2019)
dataset['yr'].astype('category').value_counts()

# Assigning string values to different months instead of numeric values
def object_map_mnths(x):
    return x.map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})

dataset[['mnth']] = dataset[['mnth']].apply(object_map_mnths)

# Displaying value counts of the 'mnth' column
dataset['mnth'].astype('category').value_counts()

# Displaying value counts of the 'holiday' column
dataset['holiday'].astype('category').value_counts()

# Assigning string values to different weekdays instead of numeric values
def str_map_weekday(x):
    return x.map({0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'})

dataset[['weekday']] = dataset[['weekday']].apply(str_map_weekday)

# Displaying value counts of the 'weekday' column
dataset['weekday'].astype('category').value_counts()

# Displaying value counts of the 'workingday' column
dataset['workingday'].astype('category').value_counts()

# Assigning string values to different weather situations instead of numeric values
# 1 = Clear, few clouds, partially cloudy
dataset.loc[(dataset['weathersit'] == 1), 'weathersit'] = 'A'

# 2 = Mist, Cloudy
dataset.loc[(dataset['weathersit'] == 2), 'weathersit'] = 'B'

# 3 = Light Snow, Heavy Rain
dataset.loc[(dataset['weathersit'] == 3), 'weathersit'] = 'C'

# Displaying value counts of the 'weathersit' column
dataset['weathersit'].astype('category').value_counts()

# 2. Data Visualisation

# Importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution of temperature
sns.distplot(dataset['temp'])

# Plotting the distribution of actual temperature
sns.distplot(dataset["atemp"])
plt.show()

# Plotting the distribution of wind speed
sns.distplot(dataset['windspeed'])
plt.show()

# Plotting the distribution of total rental bikes including both casual and registered
sns.distplot(dataset['cnt'])
plt.show()

# Converting date to datetime format
dataset['dteday'] = pd.to_datetime(dataset['dteday'])

# Selecting categorical columns
dataset_categorical = dataset.select_dtypes(exclude=['float64', 'datetime64', 'int64'])

# Displaying the names of categorical columns
dataset_categorical.columns

# Displaying the first few rows of categorical columns
dataset_categorical.head()

# Plotting boxplots to visualize the relationship between categorical variables and 'cnt'
plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
sns.boxplot(x='season', y='cnt', data=dataset)
plt.subplot(3, 3, 2)
sns.boxplot(x='mnth', y='cnt', data=dataset)
plt.subplot(3, 3, 3)
sns.boxplot(x='weekday', y='cnt', data=dataset)
plt.subplot(3, 3, 4)
sns.boxplot(x='weathersit', y='cnt', data=dataset)
plt.subplot(3, 3, 5)
sns.boxplot(x='workingday', y='cnt', data=dataset)
plt.subplot(3, 3, 6)
sns.boxplot(x='yr', y='cnt', data=dataset)
plt.subplot(3, 3, 7)
sns.boxplot(x='holiday', y='cnt', data=dataset)
plt.show()

# Converting integer columns to float
intVarlist = ["casual", "registered", "cnt"]
for var in intVarlist:
    dataset[var] = dataset[var].astype("float")

# Selecting numeric columns
dataset_numeric = dataset.select_dtypes(include=['float64'])
dataset_numeric.head()

# Plotting pairplots for numeric columns
sns.pairplot(dataset_numeric)
plt.show()

# Calculating the correlation matrix
cor = dataset_numeric.corr()
cor

# Plotting the heatmap of the correlation matrix
mask = np.tril(np.ones_like(cor, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cor, mask=mask, vmax=1, square=True, annot=True)
plt.show()

# Removing 'atemp' as it is highly correlated with 'temp'
dataset.drop("atemp", axis=1, inplace=True)

# 3. Data Preparation

# Selecting categorical columns
dataset_categorical = dataset.select_dtypes(include=['object'])
dataset_categorical.head()

# Creating dummy variables for categorical columns
dataset_dummies = pd.get_dummies(dataset_categorical, drop_first=True)
dataset_dummies.head()

# Dropping original categorical columns
dataset.drop(list(dataset_categorical.columns), axis=1, inplace=True)

# Concatenating dummy variables with the dataset
dataset = pd.concat([dataset, dataset_dummies], axis=1)
dataset.head()

# Dropping 'Instant' and 'dteday' columns
dataset.drop(['instant', 'dteday'], axis=1, inplace=True)
dataset.head()

# 4. Model Building and Evaluation

# Importing necessary libraries for model building
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import r2_score

# Splitting the dataframe into train and test datasets
np.random.seed(0)
df_train, df_test = train_test_split(dataset, train_size=0.7, test_size=0.3, random_state=100)

# Initializing MinMaxScaler
scaler = MinMaxScaler()

# Scaling the specified columns in the training set
var = ["temp", "hum", "windspeed", "casual", "registered", "cnt"]
df_train[var] = scaler.fit_transform(df_train[var])

# Displaying the statistical summary of the training set
df_train.describe()

# Checking the correlation coefficients to see which variables are highly correlated
plt.figure(figsize=(30, 30))
sns.heatmap(df_train.corr(), annot=True, cmap="YlGnBu")
plt.show()

# Splitting the training data into features (X) and target variable (Y)
x_train = df_train.drop(["casual", "registered"], axis=1)
y_train = df_train.pop('cnt')

# Adding a constant to the model
x_train_lm = sm.add_constant(x_train)

# Fitting the OLS model
lr = sm.OLS(y_train, x_train_lm).fit()

# Displaying the summary of the OLS model
lr.summary()

# Performing Recursive Feature Elimination (RFE)
lm = LinearRegression()
rfe1 = RFE(lm, 15)

# Fitting the model with 15 features
rfe1.fit(x_train, y_train)

# Displaying the selected features and their rankings
print(rfe1.support_)
print(rfe1.ranking_)

# Selecting the columns based on RFE
col1 = x_train.columns[rfe1.support_]
X_train_rfe1 = x_train[col1]

# Adding a constant to the model
x_train_rfe1 = sm.add_constant(X_train_rfe1)

# Fitting the OLS model with selected features
lm1 = sm.OLS(y_train, x_train_rfe1).fit()
lm1.summary()

# Evaluating Variance Inflation Factors (VIFs)
a = x_train_rfe1.drop('const', axis=1)
vif = pd.DataFrame()
vif['features'] = a.columns
vif['VIF'] = [variance_inflation_factor(a.values, i) for i in range(a.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by="VIF", ascending=False)
vif

# Performing RFE with 7 features
rfe2 = RFE(lm, 7)
rfe2.fit(x_train, y_train)

# Displaying the selected features and their rankings
print(rfe2.support_)
print(rfe2.ranking_)

# Selecting the columns based on RFE
col2 = x_train.columns[rfe2.support_]
x_train_rfe2 = x_train[col2]

# Adding a constant to the model
x_train_rfe2 = sm.add_constant(x_train_rfe2)

# Fitting the OLS model with selected features
lm2 = sm.OLS(y_train, x_train_rfe2).fit()
lm2.summary()

# Evaluating VIFs for the new model
b = x_train_rfe2.drop('const', axis=1)
vif1 = pd.DataFrame()
vif1['features'] = b.columns
vif1['VIF'] = [variance_inflation_factor(b.values, i) for i in range(b.shape[1])]
vif1['VIF'] = round(vif1['VIF'], 2)
vif1 = vif1.sort_values(by="VIF", ascending=False)
vif1

# Predicting 'cnt' for the training set
y_train_cnt = lm2.predict(x_train_rfe2)

# Plotting the distribution of actual vs predicted 'cnt' values
fig = plt.figure()
sns.displot((y_train, y_train_cnt), bins=20)

# Scaling the test dataset
df_test[var] = scaler.transform(df_test[var])

# Splitting the test data into features (X) and target variable (Y)
y_test = df_test.pop('cnt')
x_test = df_test.drop(["casual", "registered"], axis=1)

# Selecting the columns based on RFE for the test set
x_test_rfe2 = x_test[col2]

# Adding a constant to the test set
x_test_rfe2 = sm.add_constant(x_test_rfe2)

# Predicting 'cnt' for the test set
y_pred = lm2.predict(x_test_rfe2)

# Plotting the actual vs predicted 'cnt' values
plt.figure()
plt.scatter(y_test, y_pred)

# Calculating the R^2 score
print(r2_score(y_test, y_pred))

# Plotting the heatmap of correlations between the selected features
plt.figure(figsize=(8, 5))
sns.heatmap(dataset[col2].corr(), cmap="YlGnBu", annot=True)
plt.show()
