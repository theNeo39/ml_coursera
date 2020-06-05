import numpy as np
from gradientFunction import gradientFunction
from sigmoid import sigmoid


def gradientFunctionReg(theta, X, y, Lambda):
    m =y.size # number of training examples
    grad=np.zeros(theta.size)
    tx=np.dot(X,theta)
    cost=np.subtract(sigmoid(tx),y)
    for i in range(theta.size):
        grad[i]=np.sum(cost*X[:,i])/m
    th=(Lambda/m)*theta[1:]
    grad[1:]=grad[1:]+th
    return grad