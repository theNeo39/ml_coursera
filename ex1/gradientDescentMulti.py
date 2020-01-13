import numpy as np
from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):
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
        t2=theta[2]-((np.sum(cost*X[:,2])*alpha)/m)
        theta[0]=t0
        theta[1]=t1
        theta[2]=t2
        J_history.append(computeCostMulti(X, y, theta))
        print(J_history[i])

    return theta, J_history