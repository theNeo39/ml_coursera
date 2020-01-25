import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction
from learningCurve import learningCurve

def validationCurve(X, y, Xval, yval):
    """returns the train
    and validation errors (in error_train, error_val)
    for different values of lambda. You are given the training set (X,
    y) and validation set (Xval, yval).
    """

# Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

# You need to return these variables correctly.
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)

    for i in range(len(lambda_vec)):
        Lambda=lambda_vec[i]
        theta=trainLinearReg(X,y,Lambda)
        hx=np.dot(X,theta)
        error_train[i]=(np.sum(np.square(hx-y)))/(2*(X.shape[0]))
        hx1=np.dot(Xval,theta)
        error_val[i]=(np.sum(np.square(hx1-yval)))/(2*(Xval.shape[0]))

    return lambda_vec, error_train, error_val