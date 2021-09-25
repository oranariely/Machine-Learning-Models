#!/usr/bin/env python
# coding: utf-8

# 
# # Random Forest Project 
# 
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


loans_df = pd.read_csv('loan_data.csv')


# In[20]:


loans_df.info()


# In[21]:


loans_df.describe()


# In[22]:


loans_df.head()


# In[23]:


plt.figure(figsize=(10,6))
loans_df[loans_df['credit.policy']==0]['fico'].hist(bins=35,color='red',alpha=0.6,label='0')
loans_df[loans_df['credit.policy']==1]['fico'].hist(bins=35,color='blue',alpha=0.6,label='1')
plt.legend()


# In[32]:


plt.figure(figsize=(10,6))
loans_df[loans_df['not.fully.paid']==0]['fico'].hist(bins=35,color='red',alpha=0.6,label='0')
loans_df[loans_df['not.fully.paid']==1]['fico'].hist(bins=35,color='blue',alpha=0.8,label='1')
plt.legend()


# In[48]:


plt.figure(figsize=(10,6),tight_layout=True)
sns.countplot(x='purpose',hue='not.fully.paid',data=loans_df,palette='Set1')
# plt.xticks(rotation = 90)


# In[63]:


sns.jointplot('fico','int.rate',loans_df,color='purple')


# In[75]:


sns.lmplot(x='fico',y='int.rate',data=loans_df,hue='credit.policy',col='not.fully.paid',palette='Set1')


# In[76]:


loans_df.info()


# In[79]:


cat_feats = ['purpose']


# In[80]:


final_data = pd.get_dummies(data=loans_df, columns=cat_feats,drop_first=True)


# In[81]:


from sklearn.model_selection import train_test_split


# In[83]:


final_data.head(1)


# In[85]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[86]:


from sklearn.tree import DecisionTreeClassifier


# In[87]:


dtree = DecisionTreeClassifier()


# In[88]:


dtree.fit(X_train,y_train)


# In[89]:


pred = dtree.predict(X_test)


# In[90]:


from sklearn.metrics import confusion_matrix, classification_report


# In[96]:


print(confusion_matrix(y_test,pred), '\n\n\n', classification_report(y_test,pred))


# In[97]:


from sklearn.ensemble import RandomForestClassifier


# In[103]:


rfc = RandomForestClassifier(n_estimators=300)


# In[104]:


rfc.fit(X_train, y_train)


# In[105]:


rfc_pred = rfc.predict(X_test)


# In[107]:


print(classification_report(y_test,rfc_pred), '\n\n\n', confusion_matrix(y_test, rfc_pred))

