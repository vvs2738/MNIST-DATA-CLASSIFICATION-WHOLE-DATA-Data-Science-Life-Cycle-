
# coding: utf-8

# # IMPORTING LIBRARIES
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # READING DATA from https://www.kaggle.com/c/titanic/data

# In[2]:


tit=pd.read_csv('G:/data science skillathon/all data/titanic/train.csv')


# In[3]:


tit.info()


# In[4]:


tit.columns


# In[5]:


tit.head()


# # Exploratory Data Analysis and Data Cleaning

# In[6]:


tit['Name'].nunique()


# In[7]:


tit[tit.duplicated(subset=['Ticket', 'Fare', 'Embarked'],keep='first')].head(5)


# In[8]:


tit['Ticket'].nunique()


# In[9]:


print(tit.isnull().sum())
sns.heatmap(tit.isnull(),xticklabels=True,yticklabels=False)


# 19.8% of Age Data is missing and 77% of cabin data is missing and 0.023% of Embarked are missing

# In[10]:


#Distribution of Age Column
tit.hist(column='Age', by=None, grid=False, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False, figsize=None, layout=None, bins=10)


# As Distribution is normally distributed,  replace Age column missing values by mean

# In[11]:


#Survival based on gender category
sns.countplot(x='Survived', y=None, hue='Sex', data=tit, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, dodge=True, ax=None)


# Females survived more during Titanic sink

# In[12]:


#Survival based on class category
sns.countplot(x='Survived', y=None, hue='Pclass', data=tit, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, dodge=True, ax=None)


# Higher class people survived more during Titanic sink

# In[13]:



sns.boxplot(x='Pclass', y='Age', hue=None, data=tit, order=None, hue_order=None, orient=None, color=None, palette='summer', saturation=0.75, width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, notch=False, ax=None)


# In[14]:


sns.boxplot(x='Sex', y='Age', hue=None, data=tit, order=None, hue_order=None, orient=None, color=None, palette='summer', saturation=0.75, width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, notch=False, ax=None)


# In[15]:


def missing(c):
    Age=c[0]
    Pclass=c[1]
    if pd.isnull(Age):
            if (Pclass ==1):
                return 37
        
            elif(Pclass ==2):
                return 29
            
            else:
                return 24
    else:
        return Age
     


# In[16]:


tit['Age']=tit[['Age','Pclass']].apply(missing,axis=1)


# In[17]:


# Replacing Embarked mising value by 'UN'i.e. unknown
tit["Embarked"].fillna("un", inplace = True)


# In[18]:


# Dropping Unecessary column as ost of the values are missing 
tit.drop(columns=['Cabin','Name'],inplace=True,axis=1)


# In[19]:


tit.columns


# In[20]:


#Checking Misisng values
sns.heatmap(tit.isnull(),xticklabels=True,yticklabels=False)


# In[21]:


#One hot Encoding
pd.get_dummies(tit,columns=['Embarked']).head(10)


# In[22]:


sns.countplot(x='Sex', y=None, hue=None, data=tit, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, dodge=True, ax=None)


# In[23]:


#Count based on number of siblings/spouses aboard the Titanic
sns.countplot(x='SibSp', y=None, hue=None, data=tit, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, dodge=True, ax=None)


# In[24]:


#Count based on number of parents/children aboard the Titanic
sns.countplot(x='Parch', y=None, hue=None, data=tit, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, dodge=True, ax=None)


# In[25]:


#Survival based on number of parents with spouses/children with siblings with their spouses aboard the Titanic
sns.barplot(x="Parch", y="Survived",hue='SibSp',data=tit)


# In[26]:


#Highest Fire on Titanic Ship
plt.hist(x='Fare',data=tit)


# In[27]:


#Survival of  passenger,from location they got on the ship (C - Cherbourg, S - Southampton, Q - Queenstown)
sns.countplot(x='Survived', y=None, hue='Embarked', data=tit)

