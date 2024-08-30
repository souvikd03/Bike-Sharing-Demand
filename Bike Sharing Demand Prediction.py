#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# # 1. Data Understanding and Exploration

# In[2]:


#Import Libraries
import pandas as pd
import numpy as np


# In[3]:


#Reading the dataset
dataset = pd.read_csv("BoomBikes.csv")


# In[4]:


dataset.head()


# In[6]:


dataset.shape


# In[7]:


dataset.columns


# In[8]:


dataset.describe()


# In[9]:


dataset.info()


# In[10]:


#Assigning string values to different seasons instead of numeric values

#1 = Spring
dataset.loc[(dataset['Season'] == 1),'season'] = 'Spring'

#2 = Summer
dataset.loc[(dataset['Season'] == 2),'season'] = 'Summer'

#3 = fall
dataset.loc[(dataset['Season'] == 3),'season'] = 'fall'

#4 = winter
dataset.loc[(dataset['Season'] == 4),'season'] = 'Winter'


# In[11]:


dataset['Season'].astype('Category').value_counts()


# In[12]:


# 0 = 2018, 1 = 2019 : Year
dataset['yr'].astype('Category').value_counts()


# In[14]:


#assigning string values to different months instead of numeric values
def object_map_mnths(x):
    return x.map({1: 'Jan',2: 'Feb',3: 'Mar',4: 'Apr',5: 'May',6: 'Jun',7: 'Jul',8: 'Aug',9: 'Sep',10: 'Oct',11: 'Nov',12: 'Dec'})


# In[15]:


dataset[['mnth']] = dataset[['mnth']].apply(object_map_mnths)


# In[16]:


dataset['mnth'].astype('Category').value_counts()


# In[17]:


dataset['holiday'].astype('Category').value_counts()


# In[18]:


def str_map_weekday(x):
    return x.map({1: 'Mon',2: 'Tue',3: 'Wed',4: 'Thurs',5: 'Fri',6: 'Sat',7: 'Sun'})


# In[19]:


dataset[['weekday']] = dataset[['weekday']].apply(str_map_weekday)


# In[20]:


dataset['weekday'].astype('Category').value_counts()


# In[21]:


dataset['workingday'].astype('Category').value_counts()


# In[22]:


# 1 = Clear, few clouds, partially cloudy
dataset.loc[(dataset['weathersit'] == 1),'weathersit'] = 'A'

# 2 = Mist, Cloudy
dataset.loc[(dataset['Weathersit'] == 1),'Weathersit'] = 'B'

# 3 = Light Snow, Heavy Rain
dataset.loc[(dataset['Weathersit'] == 1),'Weathersit'] = 'C'


# In[23]:


dataset['Weathersit'].astype('Category').value_counts()


# # 2. Data Visualisation

# In[1]:


#Importing Libs
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Temperature
sns.distplot(dataset['temp'])


# In[4]:


#Actual Temperature
sns.distplot(dataset["atemp"])
plt.show()


# In[5]:


# wind speed
sns.distplot(dataset['windspeed'])
plt.show()


# In[6]:


# Target variable:count of total rental bikes including both casual and registered
sns.distplot(dataset['cnt'])
plt.show()


# In[7]:


#converting date to datetime format
dataset['dteday'] = dataset['dteday'].astype('datetime64')


# In[8]:


dataset_categorical = dataset.select_dtypes(exclude=['float64','datetime64','int64'])


# In[9]:


dataset_categorical.columns


# In[1]:


dataset_categorical


# In[5]:


plt.figure(figsize=(20,20))
plt.sub(3,3,1)
sns.boxplot(x = 'season',y = 'cnt',data=dataset)
plt.sub(3,3,2)
sns.boxplot(x = 'mnth',y = 'cnt',data=dataset)
plt.sub(3,3,3)
sns.boxplot(x = 'weekday',y = 'cnt',data=dataset)
plt.sub(3,3,4)
sns.boxplot(x = 'weathersit',y = 'cnt',data=dataset)
plt.sub(3,3,5)
sns.boxplot(x = 'workingday',y = 'cnt',data=dataset)
plt.sub(3,3,6)
sns.boxplot(x = 'yr',y = 'cnt',data=dataset)
plt.sub(3,3,7)
sns.boxplot(x = 'holiday',y = 'cnt',data=dataset)
plt.show()


# In[6]:


intVarlist = ["casual","registered","cnt"]

for var in intVarlist:
    dataset[var] = dataset[var].astype("float")


# In[7]:


dataset_numeric = dataset.select_dtypes(include=['float64'])
dataset_numeric.head()


# In[8]:


sns.pairplot(dataset_numeric)
plt.show()


# In[9]:


cor = dataset_numeric.corr()
cor


# In[10]:


#heatmap
mask = np.array(cor)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(10,10)
sns.heatmap(cor,mask=mask,vmax = 1,square = True, annot=True)


# In[11]:


#Removing atemp as it is highly correlated with temp
dataset.drop("atemp",axis=1, inplace=True)


# In[12]:


dataset.head()


# # 3. Data Preparation

# In[13]:


dataset_categorical = dataset.select_dtypes(include=['object'])


# In[14]:


dataset_categorical.head()


# In[15]:


dataset_dummies = pd.get_dummies(dataset_categorical, drop_first = True)
dataset_dummies.head()


# In[16]:


#Drop categorical variable columns
dataset = dataset.drop(List(dataset_categorical.columns),axis=1)
dataset


# In[17]:


#Concatenate dummy variables with the dataset
dataset = pd.concat([dataset,dataset_dummies], axis=1)


# In[18]:


dataset.head()


# In[19]:


dataset = dataset.drop(['Instant','dteday'],axis=1, inplace=False)
dataset.head()


# # 4. Model Building and Evaluation

# In[1]:


#Import libs
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[2]:


#Split the dataframe into train and test datasets
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(dataset,train_size = 0.7,test_size=0.3,random_state=100)


# In[3]:


df_train


# In[4]:


from sklearn.preprocessing import MinMaxScaler


# In[5]:


scaler = MinMazScaler()


# In[6]:


#Apply scalar to all columns except dummy variables
var = ["temp","hum","windspeed","casual","registered","cnt"]

df_train[var] = scaler.fit_transform(df_train[var])


# In[7]:


df_train.describe()


# In[8]:


#checking the correlation coefficients to see which variables are highly correlated
plt.figure(figsize=(30,30))
sns.heatmap(df_train.corr(), annot = True, cmap = "YlGnBu")
plt.show()


# In[9]:


#Diving into x and y
x_train = df_train.drop(["casual","registered"], axis=1)
y_train = df_train.pop('cnt')


# In[10]:


x_train.head()


# In[11]:


np.array(x_train)


# In[12]:


import statsmodels.api as sm
x_train_lm = sm.add_constant(x_train)

lr = sm.OLS(y_train,x_train_lm).fit()


# In[15]:


lr.params


# In[13]:


lm = LinearRegression()

lm.fit(x_train,y_train)


# In[14]:


print(lm.coef_)
print(im.intercept_)


# In[16]:


lr.summary()


# In[17]:


#import rfe
from sklearn.feature_selection import RFE


# In[18]:


lm = LinearRegression()
rfe1 = RFE(lm,15)

#fit with 15 feature
rfe1.fit(x_train,y_train)
print(rfe1.support_)
print(rfe1.ranking_)


# In[19]:


col1 = x_train.columns[rfe1.support_]


# In[20]:


col1


# In[21]:


X_train_rfe1 = X_train[col1]

x_train_lm = sm.add_constant(X_train_rfe1)
lm1 = sm.OLS(y_train,X_train_rfe1).fit()
lm1.summary()


# In[1]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


a = x_train_rfe1.drop('const', axis=1)


# In[5]:


# Evaluating VIFs
vif = pd.DataFrame()
vif['features'] = a.columns
vif['VIF'] = [variance_inflation_factor(a.values,i) for i in range(a.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF",ascending = False)
vif


# In[6]:


lm = LinearRegression()
rfe2 = RFE(lm,7)

#fit with 15 feature
rfe2.fit(x_train,y_train)
print(rfe2.support_)
print(rfe2.ranking_)


# In[7]:


col2 = x_train.columns[rfe2.support_]

x_train_rfe2 = x_train[col2]

x_train_rfe2 = sm.add_constant(x_train_rfe2)
lm2 = sm.OLS(y_train, x_train_rfe2).fit()
lm2.summary()


# In[8]:


# Evaluating VIFs
b = x_train_rfe2.drop('const', axis=1)
vif1 = pd.DataFrame()
vif1['features'] = b.columns
vif1['VIF'] = [variance_inflation_factor(b.values,i) for i in range(b.shape[1])]
vif1['VIF'] = round(vif['VIF'],2)
vif1 = vif1.sort_values(by = "VIF",ascending = False)
vif1


# In[9]:


y_train_cnt = lm2.predict(x_train_rfe2)


# In[10]:


fig = plt.figure()
sns.displot((y_train,y_train_cnt),bins=20)


# In[11]:


df_test[var] = scaler.transform(df_test[var])
df_test


# In[12]:


y_test = df_test.pop('cnt')
x_test = df_test.drop(["casual","registered"], axis=1)


# In[13]:


x_test.head()


# In[14]:


c = x_train_rfe2.drop('const',axis=1)


# In[15]:


col2 = c.columns


# In[16]:


x_test_rfe2 = x_test[col2]


# In[17]:


x_test_rfe2 = sm.add_constant(x_test_rfe2)


# In[18]:


x_test_rfe2.info()


# In[19]:


y_pred = lm2.predict(x_test_rfe2)


# In[20]:


plt.figure()
plt.scatter(y_test,y_pred)


# In[21]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[23]:


plt.figure(figsize=(8,5))

sns.heatmap(dataset[col2].corr(), cmap = "YlGnBu", annot = True)
plt.show()


# In[ ]:




