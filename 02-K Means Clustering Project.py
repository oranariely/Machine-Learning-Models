#!/usr/bin/env python
# coding: utf-8

# 
# ## The Data
# 
# We will use a data frame with 777 observations on the following 18 variables.
# * Private A factor with levels No and Yes indicating private or public university
# * Apps Number of applications received
# * Accept Number of applications accepted
# * Enroll Number of new students enrolled
# * Top10perc Pct. new students from top 10% of H.S. class
# * Top25perc Pct. new students from top 25% of H.S. class
# * F.Undergrad Number of fulltime undergraduates
# * P.Undergrad Number of parttime undergraduates
# * Outstate Out-of-state tuition
# * Room.Board Room and board costs
# * Books Estimated book costs
# * Personal Estimated personal spending
# * PhD Pct. of faculty with Ph.D.’s
# * Terminal Pct. of faculty with terminal degree
# * S.F.Ratio Student/faculty ratio
# * perc.alumni Pct. alumni who donate
# * Expend Instructional expenditure per student
# * Grad.Rate Graduation rate

# In[67]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


df = pd.read_csv('College_Data',index_col=0)


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[29]:


sns.scatterplot(x='Grad.Rate',y='Room.Board',data=df,hue='Private',palette='coolwarm')


# In[30]:


sns.scatterplot('F.Undergrad', 'Outstate', data=df, hue='Private')


# In[38]:


g = sns.FacetGrid(df, hue='Private', palette='coolwarm',aspect=2,size=5)
g = g.map(plt.hist,'Outstate',bins=20,alpha=.7)


# In[39]:


g = sns.FacetGrid(df, hue='Private', palette='coolwarm',aspect=2,size=5)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=.7)


# In[70]:


df[df['Grad.Rate']>100]


# In[71]:


df['Grad.Rate']['Cazenovia College'] = 100


# In[72]:


df[df['Grad.Rate']>100]


# In[73]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)


# In[74]:


kmeans.fit(df.drop('Private',axis=1))


# In[75]:


kmeans.cluster_centers_


# In[76]:


def converter(private):
    if private=='Yes':
        return 1
    else: return 0


# In[77]:


df['Cluster'] = df['Private'].apply(converter)
                   


# In[78]:


df.head()


# In[79]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))

