
# coding: utf-8

# In[21]:

import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
import matplotlib.pyplot as plt

#plot petal length and sepal width of the three types of flowers
plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()


# In[12]:

#The first 100 observations correspond to setosa and versicolor
plt.scatter(iris.data[0:100, 1], iris.data[0:100, 2], c=iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()


# In[13]:

from sklearn import svm
svc = svm.SVC(kernel='linear')
from sklearn import datasets
X = iris.data[0:100, 1:3]
y = iris.target[0:100]
svc.fit(X, y)


# In[18]:

#Adapted from https://github.com/jakevdp/sklearn_scipy2013
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()

plot_estimator(svc, X, y)
plt.show()


# In[43]:

#plotting all the 4 characteristics of versicolor which is contained in lines 51--100
plt.scatter(iris.data[51:100,0],iris.data[51:100,1], c=iris.target[51:100])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
plt.scatter(iris.data[51:100,2], iris.data[51:100,3], c=iris.target[51:100])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()


# In[44]:

#plotting all the 4 characteristics of virginica which is contained in lines 101--150
plt.scatter(iris.data[101:150,0],iris.data[101:150,1], c=iris.target[101:150])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
plt.scatter(iris.data[101:150,2], iris.data[101:150,3], c=iris.target[101:150])
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()


# In[48]:

#plotting sepal length, sepal width of all the three
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()


# In[52]:

#plotting sepal length, sepal length of all the three
plt.scatter(iris.data[:,0], iris.data[:,2], c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[2])
plt.show()


# In[53]:

#plotting sepal length, petal width of all the three
plt.scatter(iris.data[:,0], iris.data[:,2], c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[3])
plt.show()


# In[55]:

#plotting petal length, sepal width of all the three
plt.scatter(iris.data[:,1], iris.data[:,2], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()


# In[56]:

#plotting petal witdth, sepal width of all the three
plt.scatter(iris.data[:,1], iris.data[:,3], c=iris.target)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[3])
plt.show()


# In[46]:

#plotting petal length, petal width of all the three
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()


# In[47]:

#apply the SVC module to all the flowers
X = iris.data[0:, 1:3] #choose col 1 and 2 for all the rows 
y = iris.target[0:] #choose all the flowers 
svc.fit(X,y)
print(svc.fit(X,y).score(X,y)) 

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
plot_estimator(svc, X, y)
plt.show()


# In[ ]:



