import numpy as np

from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    p=np.zeros(X.shape[0])
    tx=(np.dot(theta.T,X.T)).T
    prob=sigmoid(tx)
    prob=(np.asarray(prob)).reshape(-1,1)
    for i,pr in enumerate(prob):
        if pr>=0.5:
           p[i]=1
        else:
           p[i]=0
    return p,prob