#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import urlretrieve
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

os.makedirs("data/data_preparation_phase",exist_ok=True)
os.makedirs("data/raw",exist_ok=True)

#step 3.1 - load data from the remote source
urlretrieve(url="https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv",
                       filename="dataset.csv")

shutil.move("dataset.csv","data/raw/dataset.csv")


# In[ ]:


dataset = pd.read_csv("data/raw/dataset.csv")


# In[4]:


#inspecting dataset:
dataset.head()


# In[5]:


#remove tweet_id col:
dataset = dataset.drop(columns="tweet_id")
dataset


# In[6]:


# check classes:
dataset['sentiment'].unique()


# In[7]:


#check number of samples for each class:
dataset['sentiment'].value_counts()


# In[8]:


classes = ["sadness","happiness"]


# In[9]:


modified_dataset = dataset.loc[dataset['sentiment'].isin(classes)]


# In[10]:


#verify:
modified_dataset


# In[11]:


#map classes (1 for happiness - 0 for sadness)
modified_dataset['sentiment'] = modified_dataset.loc[:,'sentiment'].apply(lambda x: classes.index(x))


# In[12]:


modified_dataset


# In[13]:


train_set,valid_set = train_test_split(modified_dataset,test_size=0.3)
valid_set,test_set = train_test_split(valid_set,test_size=0.5)


# In[14]:


#reset indices positioning:
datasets = {
    "train":train_set,
    "valid":valid_set,
    "test":test_set
}


# In[15]:


for set in datasets:
    print(f"number of samples in {set}_set is: {len(datasets[set])}")


# In[16]:


datasets


# In[17]:


for set in datasets:
    datasets[set] = datasets[set].reset_index(drop=True)


# In[ ]:


# last step in data ingestion:
for set in datasets:
    datasets[set].to_csv(f"data/data_preparation_phase/{set}_set.csv",index=False)


# In[ ]:




