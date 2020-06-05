import numpy as np


def normalEqn(X,y):
    theta=np.zeros(X.shape[1])
    
    t1=np.linalg.pinv(np.dot(X.T,X))
    t2=np.dot(t1,X.T)
    theta=np.dot(t2,y)
    return theta


