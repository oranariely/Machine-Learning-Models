#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[2]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[3]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

iris = sns.load_dataset('iris')


# In[15]:


sns.pairplot(iris,palette='Dark2',hue='species')


# In[24]:


setosa = iris[iris['species']=='setosa']
sns.kdeplot(x=setosa['sepal_length'],y=setosa['sepal_width'],cmap='plasma',shade=True )#,shade_lowest=False)


# In[25]:


from sklearn.model_selection import train_test_split


# In[28]:


X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[29]:


from sklearn.svm import SVC


# In[30]:


svc = SVC()


# In[31]:


svc.fit(X_train,y_train)


# In[32]:


pred = svc.predict(X_test)


# In[34]:


from sklearn.metrics import confusion_matrix, classification_report


# In[38]:


print(confusion_matrix(y_test,pred))


# In[37]:


print(classification_report(y_test,pred))


# In[43]:


from sklearn.model_selection import GridSearchCV


# In[44]:


param_grid = {'C': [0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}


# In[53]:


grid_search = GridSearchCV(SVC(),param_grid,verbose=2)
grid_search.fit(X_train,y_train)


# In[55]:


pred = grid_search.predict(X_test)


# In[57]:


print(confusion_matrix(y_test,pred))


# In[58]:


print(classification_report(y_test,pred))

