#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation

# Loading the dependencies
# 

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", message=".*memory leak on Windows with MKL.*")


# Data Pre-processing

# In[2]:


customer_data = pd.read_csv('Mall_Customers.csv')


# In[3]:


customer_data.head()


# In[4]:


customer_data.shape


# In[5]:


customer_data.info()


# In[6]:


customer_data.isnull().sum()


# Customer Segmentation 

# In[7]:


X = customer_data.iloc[:,[3,4]].values
print(X)


# In[12]:


# Select the features for clustering
X = customer_data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Initialize an empty list to store the Within-Cluster Sum of Squares (WCSS)
wcss = []

# Fit KMeans for different values of clusters and compute the WCSS
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[9]:


# Plot the elbow method to determine the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[10]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0, n_init=10)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print(Y)


# In[11]:


plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[ ]:




