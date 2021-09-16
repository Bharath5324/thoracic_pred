#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('./thoracic-data.csv')
# data = pd.DataFrame(data=rand.drop('id', index=1), index=rand['id'])
data


# In[3]:


data.index = data['id']
data.drop('id', axis = 1, inplace=True)


# In[4]:


data


# In[5]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[6]:


data['DGN'] = le.fit_transform(data['DGN'])

# data['PRE4'] = le.fit_transform(data['PRE4'])

# data['PRE5'] = le.fit_transform(data['PRE5'])

data['PRE6'] = le.fit_transform(data['PRE6'])

data['PRE7'] = le.fit_transform(data['PRE7'])

data['PRE8'] = le.fit_transform(data['PRE8'])

data['PRE9'] = le.fit_transform(data['PRE9'])

data['PRE10'] = le.fit_transform(data['PRE10'])

data['PRE11'] = le.fit_transform(data['PRE11'])

data['PRE14'] = le.fit_transform(data['PRE14'])

data['PRE17'] = le.fit_transform(data['PRE17'])

data['PRE19'] = le.fit_transform(data['PRE19'])

data['PRE25'] = le.fit_transform(data['PRE25'])

data['PRE30'] = le.fit_transform(data['PRE30'])

data['PRE32'] = le.fit_transform(data['PRE32'])

data['Risk1Yr'] = le.fit_transform(data['Risk1Yr'])


# In[7]:


data['Risk1Yr'].value_counts()


# In[8]:


y = data['Risk1Yr']
x = data.drop(['Risk1Yr'], axis = 1)
x


# In[9]:


y


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[12]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[13]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 1 )


# In[14]:


classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


# In[15]:


y_pred = classifier.predict(x_test)
from sklearn import metrics

print(metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

conf_mat = metrics.confusion_matrix(y_test, y_pred)
TP = conf_mat[0,0]
FP = conf_mat[1,0]
TN = conf_mat[1,1]
FN = conf_mat[0,1]
print(conf_mat)

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
precision = TP/(TP+FP)
pred_val = TN/(TN + FN)
accuracy = (TP + FN)/ (TP + TN + FP + FN)

print("sensitivity: ", sensitivity)
print("Precision: ", precision)
print("Specificity: ", specificity)
print("Predicted Value: ", pred_val)
print("Accuracy: ", accuracy)
print("Precision Score: ", metrics.precision_score(y_test, y_pred))
print("Recall Score: ", metrics.recall_score(y_test, y_pred))
print("F1 Score: ", metrics.f1_score(y_test, y_pred))


# In[16]:


from os import system, name 
def clear():
    if name == 'nt': 
        _ = system('cls') 
    else: 
        _ = system('clear') 


# In[17]:
clear()

def userinp():
    put = []
    put.append(input("Please enter the disease your diagnosed wth from the following options: \nDGN3, DGN2, DGN4, DGN6, DGN5, DGN8, DGN1. \n").upper())
    clear()
    put.append(float(input("Please enter your forced vital capacity:\n")))
    clear()
    put.append(float(input("Please enter the value of you FEV1:\n")))
    clear()
    put.append(input("Please enter the disease your Performance Status from the following options: \nPRZ2, PRZ1, PRZ0. \n").upper())
    clear()
    put.append(input("Please enter 'T' if you've experienced any pain before surgery andd 'F' if not\n").upper())
    clear()
    put.append(input("Please enter 'T' if you've experienced any haemoptysis before surgery andd 'F' if not\n").upper())
    clear()
    put.append(input("Please enter 'T' if you've experienced any Dyspnoea before surgery andd 'F' if not\n").upper())
    clear()
    put.append(input("Please enter 'T' if you've had any cough before surgery andd 'F' if not\n").upper())
    clear()
    put.append(input("Please enter 'T' if you've had any weakness before surgery andd 'F' if not\n").upper())
    clear()
    put.append(input("Please enter the size of the tumour from the following options: \nOC11, OC12, OC13, OC14(With OC11 being the samllest and OC14 the largest )\n").upper())
    clear()
    put.append(input("Please enter 'T' if you've had any diabetes mellitus(DM) before surgery andd 'F' if not\n").upper())
    clear()
    put.append(input("Please enter 'T' if you've had any MI (Myocardial Infarction) before surgery andd 'F' if not\n").upper())
    clear()
    put.append(input("Please enter 'T' if you've had any peripheral arterial diseases  before surgery andd 'F' if not\n").upper())
    clear()
    put.append(input("Please enter 'T' if you've a habit of smoking and 'F' if not\n").upper())
    clear()
    put.append(input("Please enter 'T' if you've asthma and 'F' if not\n").upper())
    clear()
    put.append(input("Please enter your age:\n"))
    clear()
    temp = pd.DataFrame(put,index = x.columns).transpose()
    dat = pd.read_csv('./thoracic-data.csv').drop(['id', 'Risk1Yr'], axis = 1)
    for i in list(dat.columns):
        if i != 'AGE' and i != 'PRE4' and i != 'PRE5':
            le.fit(dat[i])
            temp[i] = le.transform(temp[i])
    if classifier.predict(temp) == 0:
        print("You've been tested negative of the risk in dying within one year")
    else:
        print("You've been tested negative of the risk in dying within one year")


# In[18]:


userinp()

