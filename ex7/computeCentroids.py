import numpy as np


def computeCentroids(X, idx, K):
    """returns the new centroids by
    computing the means of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    """

# Useful variables
    m, n = X.shape

# You need to return the following variables correctly.
    centroids =np.zeros((K,X.shape[1]))
    for i in range(K):
        centroids[i]=np.mean(X[np.where(idx==i),:].squeeze(),axis=0)
    return centroids
