from sigmoid import sigmoid
import numpy as np


def gradientFunction(theta, X, y):
    m =y.size # number of training examples
    grad=np.zeros(theta.size)
    tx=np.dot(X,theta)
    cost=np.subtract(sigmoid(tx),y)
    for i in range(theta.size):
        grad[i]=np.sum(cost*X[:,i])/m
    return grad
