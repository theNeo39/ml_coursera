import numpy as np

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

def learningCurve(X, y, Xval, yval, Lambda):
    """returns the train and
    cross validation set errors for a learning curve. In particular,
    it returns two vectors of the same length - error_train and
    error_val. Then, error_train(i) contains the training error for
    i examples (and similarly for error_val(i)).

    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.
    """
    error_train = np.zeros(X.shape[0])
    error_val   = np.zeros(X.shape[0])
    for i in range(1,(X.shape[0]+1)):
        theta=trainLinearReg(X[:i,:],y[:i],Lambda)
        hx=np.dot(X[:i,:],theta)
        error_train[i-1]=(np.sum(np.square(hx-y[:i])))/(2*(i))
        hx1=np.dot(Xval,theta)
        error_val[i-1]=(np.sum(np.square(hx1-yval)))/(2*(Xval.shape[0]))
    return error_train, error_val