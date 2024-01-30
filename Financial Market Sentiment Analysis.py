#!/usr/bin/env python
# coding: utf-8

# # Financial Market News - Sentiment Analysis
# 
# This is a data (dummy) of Financial Market Top 25 News for the Day and Task is to Train and Predict Model for Overall Sentiment Analysis

# # Import Library

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# # Import Dataset

# In[3]:


df = pd.read_csv(r'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Financial%20Market%20News.csv', encoding = "ISO-8859-1")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.columns


# # Get Feature Selection

# In[8]:


' '.join(str(x) for x in df.iloc[1,2:27])


# In[9]:


df.index


# In[10]:


len(df.index)


# In[11]:


news = []
for row in range(0,len(df.index)):
    news.append(' '.join(str(x) for x in df.iloc[row,2:27]))


# In[12]:


type(news)


# In[13]:


news[0]


# In[14]:


X = news


# In[15]:


type(X)


# # Get Feature Text Conversion to Bag of Words

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer


# In[17]:


cv = CountVectorizer(lowercase = True, ngram_range=(1,1))


# In[18]:


X = cv.fit_transform(X)


# In[19]:


X.shape


# In[20]:


y = df['Label']


# In[21]:


y.shape


# # Get Train Test Split

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 2529)


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


rf = RandomForestClassifier(n_estimators=200)


# In[26]:


rf.fit(X_train, y_train)


# In[27]:


y_pred = rf.predict(X_test)


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[29]:


confusion_matrix(y_test, y_pred)


# In[30]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




