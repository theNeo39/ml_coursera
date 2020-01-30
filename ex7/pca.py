import numpy as np


def pca(X):
    """computes eigenvectors of the covariance matrix of X
      Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """

    # Useful values
    m, n = X.shape
    sigma=np.dot(X.T,X)/m
    U,S,V=np.linalg.svd(sigma)
    return U,S