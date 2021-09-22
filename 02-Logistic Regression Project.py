#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


ad_data = pd.read_csv("advertising.csv")


# In[5]:


ad_data.head()


# In[6]:


ad_data.info()


# In[7]:


ad_data.describe()


# In[25]:


sns.set_style(style='whitegrid')
sns.histplot(ad_data['Age'],bins=30)
# plt.xlim((10,70))


# In[27]:


sns.jointplot(x='Age',y='Area Income',data=ad_data)


# In[39]:


sns.jointplot(data=ad_data, x='Age', y='Daily Time Spent on Site',kind='kde',color='red')


# In[47]:


sns.jointplot(data=ad_data,x='Daily Time Spent on Site', y='Daily Internet Usage')


# In[49]:


sns.pairplot(ad_data,hue='Clicked on Ad')


# In[56]:


ad_data.describe()


# In[62]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[58]:


X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[66]:


logmodel = LogisticRegression()
logmodel.fit(X=X_train, y=y_train)


# In[67]:


pre = logmodel.predict(X_test)


# In[72]:


from sklearn.metrics import classification_report, confusion_matrix


# In[71]:


print(classification_report(y_test,pre))


# In[74]:


print(confusion_matrix(y_test,pre))

