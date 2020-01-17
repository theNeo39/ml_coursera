import sys
sys.path.insert(1, 'D:\GitProjects\Machine Learning\ML Coursera\ex2')
from costFunctionReg import costFunctionReg

def lrCostFunction(theta, X, y, Lambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    J,grad=costFunctionReg(theta,X,y,Lambda)
    return J,grad
