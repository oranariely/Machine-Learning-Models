#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv("Ecommerce Customers")


# In[6]:


customers.describe()


# In[4]:


customers.head()


# In[7]:


customers.info()


# In[18]:


sns.jointplot(data=customers,x="Time on Website",y="Yearly Amount Spent")


# In[19]:


sns.jointplot(data=customers,x="Time on App",y="Yearly Amount Spent")


# In[27]:


sns.jointplot(data=customers,x='Time on App',y='Length of Membership',kind='hex',color='grey')


# In[28]:


sns.pairplot(data=customers)


# In[32]:


sns.lmplot(data=customers,x='Length of Membership', y='Yearly Amount Spent')


# In[35]:


X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = customers['Yearly Amount Spent']


# In[36]:


X.head()


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


model = LinearRegression()


# In[41]:


model.fit(X=X_train,y=y_train)


# In[42]:


model.coef_


# In[44]:


predictions = model.predict(X_test)


# In[52]:


plt.scatter(x=y_test,y=predictions)
plt.xlabel('y_test')
plt.ylabel('prediction')


# In[60]:


from sklearn import metrics
print(' MAE' ,metrics.mean_absolute_error(y_true=y_test,y_pred=predictions), '\n',
'MSE', metrics.mean_squared_error(y_true=y_test,y_pred=predictions), '\n',
'RMSE' ,np.sqrt(metrics.mean_squared_error(y_true=y_test,y_pred=predictions)))


# In[66]:


sns.histplot(data=customers,x=(y_test-predictions),bins=50,kde=True,alpha=0.4)


# In[68]:


df = pd.DataFrame(model.coef_,X.columns,columns=["Coeffecient"])


# In[69]:


df

