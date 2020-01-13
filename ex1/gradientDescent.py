import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples
    
    for i in range(num_iters):
        
        cost=np.subtract(np.dot(theta.T,X.T).T,y)
        t0=theta[0]-((np.sum(cost)*alpha)/m)
        t1=theta[1]-((np.sum(cost*X[:,1])*alpha)/m)
        theta[0]=t0
        theta[1]=t1
        J_history.append(computeCost(X, y, theta))
    return theta, J_history
   