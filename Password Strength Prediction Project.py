#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv(r'C:\Users\ralfy\OneDrive\Desktop\CSV Files\data.csv',error_bad_lines=False)
data.head()


# In[4]:


data['strength'].unique()


# In[5]:


data.isna().sum()


# In[6]:


data[data['password'].isnull()]


# In[7]:


data.dropna(inplace=True)


# In[8]:


data.isnull().sum()


# In[9]:


sns.countplot(data['strength'])


# In[10]:


password_tuple=np.array(data)


# In[11]:


password_tuple


# In[12]:


import random
random.shuffle(password_tuple)


# In[13]:


x=[lables[0] for lables in password_tuple]
y=[lables[1] for lables in password_tuple]


# In[14]:


x


# In[15]:


y


# In[16]:


def word_divide_char(inputs):
    character=[]
    for i in inputs:
        character.append(i)
    return character


# In[17]:


word_divide_char('kzde5577')


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[19]:


vectorizer=TfidfVectorizer(tokenizer=word_divide_char)


# In[20]:


X=vectorizer.fit_transform(x)


# In[21]:


X.shape


# In[22]:


vectorizer.get_feature_names()


# In[23]:


first_document_vector=X[0]
first_document_vector


# In[24]:


first_document_vector.T.todense()


# In[25]:


df=pd.DataFrame(first_document_vector.T.todense(),index=vectorizer.get_feature_names(),columns=['TF-IDF'])
df.sort_values(by=['TF-IDF'],ascending=False)


# In[26]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[29]:


X_train.shape


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[32]:


clf=LogisticRegression(random_state=0,multi_class='multinomial')


# In[33]:


clf.fit(X_train,y_train)


# In[34]:


dt=np.array(['%@123abcd'])
pred=vectorizer.transform(dt)
clf.predict(pred)


# In[35]:


y_pred=clf.predict(X_test)
y_pred


# In[36]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[37]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))


# In[39]:


from sklearn.metrics import classification_report


# In[40]:


print(classification_report(y_test,y_pred))

