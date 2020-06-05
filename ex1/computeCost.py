import numpy as np

def computeCost(X, y, theta):
    m = y.size
    J = 0
    x1=np.subtract(np.dot(X,theta),y)
    J=np.sum(np.square(x1))/(2*m)
    return J


