from sigmoid import sigmoid
import numpy as np


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    m =y.size # number of training examples
    grad=np.zeros(theta.size)
    tx=(np.dot(theta.T,X.T)).T
    cost=np.subtract(sigmoid(tx),y)
    for i in range(theta.size):
        grad[i]=np.sum(cost*X[:,i])/m
    return grad
