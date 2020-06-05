import numpy as np
from numpy import log
from sigmoid import sigmoid
from gradientFunction import gradientFunction

def costFunction(theta, X,y):
# Initialize some useful values
    m = y.size # number of training examples
    tx=np.dot(X,theta)
    t1=(-1*y)*(np.log(sigmoid(tx)))
    t2=(1-y)*(np.log(1-sigmoid(tx)))
    t3=np.subtract(t1,t2)
    J=(np.sum(t3))/m
    grad=gradientFunction(theta,X,y)
    return J,grad
