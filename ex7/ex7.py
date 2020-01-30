## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ================= Part 1: Find Closest Centroids ====================
import numpy as np
import scipy.io
import imageio as im
import matplotlib.pyplot as plt
import seaborn as sns
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids


print('Finding closest centroids.')

# Load an example dataset that we will be using
data = scipy.io.loadmat('ex7data2.mat')
X = data['X']

#sns.scatterplot(X[:,0],X[:,1])

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the
# initial_centroids
idx = findClosestCentroids(X, initial_centroids)
idx=idx.astype(int)
print('Closest centroids for the first 3 examples:')
print(idx[0:3])
print('(the closest centroids should be 0, 2, 1 respectively)')

## ===================== Part 2: Compute Means =========================

print('Computing centroids means.')

#  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)
print('Centroids computed after initial finding of closest centroids:')
for c in centroids:
    print(c)

print('(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]')

## =================== Part 3: K-Means Clustering ======================

print('Running K-Means clustering on example dataset.')

# Load an example dataset
data = scipy.io.loadmat('ex7data2.mat')
X = data['X']

# Settings for running K-Means
K = 3
max_iters = 10


initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
print('K-Means Done.')
## ============= Part 4: K-Means Clustering on Pixels ===============

print('Running K-Means clustering on pixels from an image.')

#  Load an image of a bird
A = im.imread('bird_small.png')
M=A.copy()

A = A / 255.0 # Divide by 255 so that all values are in the range 0 - 1

X = A.reshape(-1, 3)
K = 16 
max_iters = 10

initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters)
## ================= Part 5: Image Compression ======================
print('Applying K-Means to compress an image.')
temp=np.zeros((X.shape[0],X.shape[1]))
idx=idx.astype(int)
for i,e in enumerate(idx.reshape(-1,1)):
    temp[i]=centroids[e,:]

X_recovered=temp.reshape(A.shape)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow((A*255).astype(np.uint8))
ax[0].set_title('Original')
ax[0].grid(False)

# Display compressed image, rescale back by 255
ax[1].imshow((X_recovered*255).astype(np.uint8))
ax[1].set_title('Compressed, with %d colors' % K)
ax[1].grid(False)