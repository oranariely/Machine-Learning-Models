#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('KNN_Project_Data')


# In[4]:


df.head()


# In[6]:


sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


scaler = StandardScaler()


# In[17]:


scal_feat = scaler.transform(X)


# In[19]:


df_feat = pd.DataFrame(scal_feat,columns=df.columns[:-1])
df_feat.head()


# In[21]:


from sklearn.model_selection import train_test_split


# In[23]:


X = df_feat
X_train, X_test, y_train, y_test = train_test_split(df_feat, y, test_size=0.3, random_state=101)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[29]:


pred = knn.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report, confusion_matrix


# In[32]:


print(confusion_matrix(y_test,pred))


# In[33]:


print(classification_report(y_test,pred))


# In[37]:


error_rate = []

for i in range(1,40):
    knn_i = KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(X_train,y_train)
    pred_i = knn_i.predict(X_test)
    error_rate.append(np.mean((pred_i != y_test)))


# In[60]:


plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
plt.plot(range(1,40),error_rate,ls='--', marker='o',markerfacecolor='red',markersize=10)
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.title('Error Rate vs K-Neighbors')


# In[63]:


knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)


# In[64]:


print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

