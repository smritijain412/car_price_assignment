#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install seaborn')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


data=pd.read_csv('cars_price.csv')


# In[8]:


data.head()


# In[9]:


#check statistical info
data.describe()


# # clean data 

# In[10]:


# check null values in data det
data.isna().sum().values


# In[12]:


# very few values are null. rather than replace these values with 0 or mean method we can drop it 
data.dropna(axis=0,inplace=True)


# In[13]:


# drop unneccessary column from data frame and make copy it.
data1=data.drop(labels=['Unnamed: 0'],axis=1)


# In[14]:


#checek duplicat values
data1.duplicated().sum()


# In[15]:


# drop these 
data1.drop_duplicates(inplace=True)
data1


# In[ ]:


#plot graph between dependent and independent variables 


# In[16]:


#Graph price vs mileage 
plt.figure(figsize=(11.7,8.7))
sns.relplot(x='mileage(kilometers)',y='priceUSD',sizes=(15,200),data=data1)


# In[17]:


# Plotting a Histogram
data1.make.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title('Number of cars by make')
plt.ylabel('Number of cars')
plt.xlabel('Make')


# In[ ]:


#highest number of cars are volkswegan


# In[19]:


#2 price vs condition
sns.scatterplot(data1['condition'],data1['priceUSD'])


# In[20]:


#year vs priceUSD
sns.lineplot(data1['year'],data1['priceUSD'])


# In[29]:


sns.barplot(data1['fuel_type'],data1['year'])


# In[23]:


# plot data
def groupby(x1,x2):
    fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
    data1.groupby([x1,x2]).count()['priceUSD'].unstack().plot(ax=ax)


# In[24]:


groupby('year','transmission')


# In[ ]:


#here we can see mechanic cars are rapidly grow in between 1990 to 2000 and after 2000 to 2010 its fall down but its more than auto.
#where auto cars are grow (less than mechanics)in middle of 1990 to 2010.But 2010 to 2020 its decrease. After 2010 to 2020 auto cars are more than mechanics.


# In[22]:


groupby('mileage(kilometers)','fuel_type')


# In[26]:


groupby('year','fuel_type')


# In[ ]:


#convert categorical data into numeric 


# In[29]:


data1['fuel_type'].replace(to_replace=['petrol','diesel'],value=[0,1],inplace=True)


# In[30]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
data1["make"] = lb_make.fit_transform(data1["make"])


# In[31]:


data['condition'].value_counts()


# In[32]:


#You can see here "with mileage" car has more than other. We can use if else situation where with mileage 
#has 1 and other car has 0 value
data1["condition_code"] = np.where(data1["condition"].str.contains("with mileage"),1,0)


# In[33]:


data1['drive_unit'].value_counts()


# In[34]:


data1['transmission'].value_counts()


# In[35]:


data1['transmission'].replace(to_replace=['mechanics','auto'],value=[0,1],inplace=True)


# In[41]:


data1.info()


# In[37]:


X=data1.iloc[:,[3,5,6,7,9,12]]
Y=data1.iloc[:,[2]]


# In[39]:


from sklearn.feature_selection import chi2


# In[40]:


chi2(X,Y)


# In[41]:


#now bases on study and servey we can do hit and trial, opt those variables which are important in car price prediction ""
df=data1.iloc[:,[0,2,3,5,6,7,9,12]]


# In[43]:


df.info()


# In[42]:


df.corr()


# In[46]:


from sklearn.model_selection import train_test_split


# In[43]:


# split data set into X and Y
X1=df.iloc[:,[2,4,5,6,7]]
Y1=df.iloc[:,1]


# In[44]:


chi2(X1,Y1)


# In[47]:


x_train,x_test,y_train,y_test=train_test_split(X1,Y1)


# In[48]:


print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[49]:


from sklearn.linear_model import Lasso,ridge_regression,ridge,LogisticRegression


# In[50]:


#create function 
def models(model):
    model.fit(x_train,y_train)
    score=model.score(x_test,y_test)
    return score


# In[51]:


lasso=Lasso()
models(lasso)


# In[69]:


lr=LogisticRegression()
models(lr)


# In[52]:


from sklearn.ensemble import ExtraTreesRegressor


# In[57]:


tree=ExtraTreesRegressor()


# In[58]:


models(tree)


# In[ ]:


#2 Try with different values and convert categorical into numerical and apply ensemble model 


# In[59]:


data_cat=data.loc[:,['condition','fuel_type','transmission','drive_unit','priceUSD']]


# In[60]:


data_cat['fuel_type'].replace(to_replace=['petrol','diesel'],value=[0,1],inplace=True)
data_cat['transmission'].replace(to_replace=['mechanics','auto'],value=[0,1],inplace=True)


# In[61]:


data_cat=pd.get_dummies(data_cat,drop_first=True)


# In[62]:


y=data_cat['priceUSD']
y.shape


# In[63]:


data_cat=data_cat.drop(labels=['condition_with damage','priceUSD'],axis=1)


# In[64]:


X_train,X_test,Y_train,Y_test=train_test_split(data_cat,y)


# In[65]:


tree.fit(X_train,Y_train)
tree.score(X_test,Y_test)


# In[ ]:




