
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from scipy import special, optimize
from scipy.cluster.vq import vq, kmeans, whiten


# In[2]:

df = pd.read_csv('/home/nitcob/Downloads/un.csv')


# In[3]:

df = df.fillna(0)


# In[4]:

x = df['lifeMale']


# In[5]:

y = df['lifeFemale']


# In[6]:

z = df['infantMortality']


# In[7]:

a = df['GDPperCapita']


# In[8]:

plt.scatter(x, y)
plt.show()


# In[9]:

X = np.array([x, y, z, a])


# In[10]:

kmeans =KMeans(n_clusters= 4)


# In[11]:

kmeans.fit(X)


# In[12]:

#Center marker of the clusters
centroids = kmeans.cluster_centers_


# In[13]:

labels = kmeans.labels_


# In[14]:

print(centroids)
print(labels)


# In[15]:

#graphing the kmeans algorith


# In[16]:

colors =["g.","r.","b.","y."]
for i in range(len(X)):
    print("coordinate':",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], X[i][2], X[i][3], colors[labels[i]], markersize=10)
    
    


# In[21]:

plt.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], centroids[:, 3], marker = "x",linewidths = 5, zorder = 10)


# In[ ]:

plt.show()


# In[ ]:



