import numpy as np

from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    p=(np.asarray(np.zeros(X.shape[0]))).reshape(-1,1)
    print(p.shape)
    tx=np.dot(theta.T,X.T).T
    prob=sigmoid(tx)
    prob=(np.asarray(prob)).reshape(-1,1)
    print(prob.shape)
    for i,pr in enumerate(prob):
        if pr>=0.5:
           p[i]=1
        else:
           p[i]=0

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters.
#               You should set p to a vector of 0's and 1's
#


# =========================================================================

    return p,prob