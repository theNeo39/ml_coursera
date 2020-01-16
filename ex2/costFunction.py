import numpy as np
from numpy import log
from sigmoid import sigmoid
from gradientFunction import gradientFunction

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

# Initialize some useful values
    m = y.size # number of training examples
    tx=(np.dot(theta.T,X.T)).T
    t1=(-1*y)*(np.log(sigmoid(tx)))
    t2=(1-y)*(np.log(1-sigmoid(tx)))
    t3=np.subtract(t1,t2)
    J=(np.sum(t3))/m
    grad=gradientFunction(theta,X,y)
    return J,grad
