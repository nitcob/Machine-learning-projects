
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


# In[2]:

df = pd.read_csv('/home/nitcob/Downloads/ideal_weight.csv')


# In[3]:

df.rename(columns={"'id'": "id", "'sex'": "sex", "'sex'": 'sex', "'actual'": "actual", "'ideal'": "ideal", "'diff'": "diff"}, inplace=True)


# In[4]:

df['sex'] = df['sex'].map(lambda x: x.rstrip("'").strip("'"))


# In[5]:

#use the two columns as series that allows the graph to work better
plt.hist([df.ideal, df.actual])


# In[6]:

plt.hist(df.actual, bins = 10, alpha=0.3, label='actual')
plt.hist(df.ideal, bins = 10, alpha=0.3, label='ideal')
plt.title('Actual And Ideal Weights')
plt.xlabel('Weight')
plt.ylabel('count')
plt.legend(loc='upper right')
plt.show()


# In[7]:

pd.value_counts(df.sex, sort=False)


# In[8]:

# Gaussian Naive Bayes
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import csv


# In[18]:

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# load the iris datasets
dataset = df[['actual', 'ideal', 'diff']]
# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(df[['actual', 'ideal', 'diff']], df['sex'])
print(model)
# # make predictions
expected = df['sex']
predicted = model.predict(df[['actual', 'ideal', 'diff']])
# # summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[ ]:




# In[ ]:



