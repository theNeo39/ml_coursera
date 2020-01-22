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
        t=np.zeros(X.shape[1])
        for j in range(X.shape[1]):
         t[j]=theta[j]-((np.sum(cost*X[:,j])*alpha)/m)
        for j in range(X.shape[1]):
         theta[j]=t[j]
        J_history.append(computeCostMulti(X, y, theta))
        print(J_history[i])

    return theta, J_history