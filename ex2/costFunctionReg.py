import numpy as np
from costFunction import costFunction
from sigmoid import sigmoid
from gradientFunctionReg import gradientFunctionReg


def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
       # number of training examples
    m = y.size # number of training examples
    tx=(np.dot(theta.T,X.T)).T
    t1=(-1*y)*(np.log(sigmoid(tx)))
    t2=(1-y)*(np.log(1-sigmoid(tx)))
    t3=np.subtract(t1,t2)
    tj=(np.sum(t3))/m
    th=theta[1:]
    tsq=np.square(th)
    tsum=np.sum(tsq)
    tsum=(Lambda*tsum)/(2*m)
    J=tj+tsum
    grad=gradientFunctionReg(theta,X,y,Lambda)
    return J,grad
