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
    grad[0]=np.sum(cost*X[:,0])/m
    grad[1]=np.sum(cost*X[:,1])/m
    grad[2]=np.sum(cost*X[:,2])/m
    
    
   
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta


# =============================================================

    return grad
