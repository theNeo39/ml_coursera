import numpy as np
def computeCostMulti(X, y, theta):
    """
     Compute cost for linear regression with multiple variables
       J = computeCost(X, y, theta) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y
    """
    m = y.size
    J = 0
    x1=np.subtract((np.dot(theta.T,X.T).T),y)
    J=np.sum(np.square(x1))/(2*m)
    return J
