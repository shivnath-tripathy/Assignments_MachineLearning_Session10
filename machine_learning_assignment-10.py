
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets


# In[2]:


iris = datasets.load_iris()

print(iris.DESCR)


# In[3]:


irisData = iris.data
print("Dimensions:")
print(irisData.shape)
print("---")
print("First 5 samples:")
print(irisData[:5,:])
print("---")
print("Feature names:")
print(iris.feature_names)


# In[4]:


pca =decomposition.PCA(n_components=3) # two components
pca.fit(irisData) # run PCA, putting in raw version for fun
print("Principal components:")
print(pca.components_)
print("---")
print("Compressed - 4D to 3D:")
print(pca.transform(irisData)[:6,:3]) # first 5 obs
print("---")
print("Reconstructed - 3D to 4D:")
print(pca.inverse_transform(pca.transform(irisData))[:5,:]) # first 5 obs


# In[5]:


get_ipython().magic('matplotlib inline')
y = iris.target
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = decomposition.PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[ ]:


ax1 = plt.axes(projection='3d')


# In[6]:




# Data for three-dimensional scattered points
zdata =X_reduced[:, 2]
xdata = X_reduced[:, 0]
ydata = X_reduced[:, 1]
ax1.scatter3D(xdata, ydata, zdata, c=y);

