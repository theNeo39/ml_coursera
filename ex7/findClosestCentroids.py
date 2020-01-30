import numpy as np


def findClosestCentroids(X, centroids):
    """returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

# Set K
    K = len(centroids)
    c=np.zeros(K)
    tx=np.zeros((1,X.shape[1]))

# You need to return the following variables correctly.
    idx = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(K):
            tx=X[i,:]-centroids[j,:]
            c[j]=np.sum(np.square(tx))
        idx[i]=np.argmin(c)
    return idx

