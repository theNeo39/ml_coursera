import numpy as np


def linearRegCostFunction(theta,X, y, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    m=y.size
    grad=np.zeros(theta.shape[0])
    hx=np.dot(X,theta)
    J_temp=(np.sum(np.square(hx-y)))/(2*m)
    reg=(Lambda/(2*m))*(np.sum(np.square(theta[1:])))
    J=J_temp+reg
    t=np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        if(i==0):
            grad[i]=np.sum((hx-y))/m
        else:
            grad[i]=(np.sum((hx-y)*X[:,i]))/m+ (Lambda/m)*theta[i]
            
        
    return J,grad