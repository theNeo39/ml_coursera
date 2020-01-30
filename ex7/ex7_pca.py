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
#     projectData.m`
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import seaborn as sns
from featureNormalize import featureNormalize
from pca import pca
from projectData import projectData
from recoverData import recoverData
from kMeansInitCentroids import kMeansInitCentroids
from runkMeans import runkMeans
import imageio as im
from plotDataPoints import plotDataPoints
import sys
sys.path.insert(1, 'D:\GitProjects\Machine Learning\ML Coursera\ex3')
from displayData import displayData

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize

print('Visualizing example dataset for PCA.')
#  The following command loads the dataset. You should now have the 
#  variable X in your environment
data = scipy.io.loadmat('ex7data1.mat')
X = data['X']

#  Visualize the example dataset
sns.scatterplot(X[:,0],X[:,1])


## =============== Part 2: Principal Component Analysis ===============

print('Running PCA on example dataset.')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

#  Compute mu, the mean of the each feature
#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=0.25)

for i in range(2):
    ax.arrow(mu[0], mu[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i],
             head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

ax.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
ax.grid(False)

print('Top eigenvector: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
print(' (you should expect to see [-0.707107 -0.707107])')

## =================== Part 3: Dimension Reduction ===================
print('Dimension reduction on example dataset.')

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: %f', Z[0])
print('this value should be about 1.481274')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: %f %f'% (X_rec[0, 0], X_rec[0, 1]))
print('this value should be about  -1.047419 -1.047419')

#  Draw lines connecting the projected points to the original points
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo', ms=8, mec='b', mew=0.5)
ax.set_aspect('equal')
ax.grid(False)
plt.axis([-3, 2.75, -3, 2.75])

# Draw lines connecting the projected points to the original points
ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', mec='r', mew=2, mfc='none')
for xnorm, xrec in zip(X_norm, X_rec):
    ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('Loading face dataset.')

#  Load Face dataset
data = scipy.io.loadmat('ex7faces.mat')
X = data['X']

#  Display the first 100 faces in the dataset
displayData(X[:100, :])
 

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('Running PCA on face dataset.\n(this might take a minute or two ...)\n\n')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S= pca(X_norm)

#  Visualize the top 36 eigenvectors found
displayData(U[:,:36].T)


## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('Dimension reduction for face dataset.')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print('%d %d'% Z.shape)

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.')

K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
displayData(X_norm[:100,:])
plt.gcf().suptitle('Original faces')

# Display reconstructed data from only k eigenfaces
displayData(X_rec[:100,:])
plt.gcf().suptitle('Recovered faces')