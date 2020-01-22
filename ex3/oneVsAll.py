import numpy as np
from scipy import optimize
from lrCostFunction import lrCostFunction
#from ex2.gradientFunctionReg import gradientFunctionReg


def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

# Some useful variables
    m, n = X.shape

# You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

# Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
 
 # Set Initial theta
    initial_theta = np.zeros(n + 1)
    options = {'maxiter': 10}
    for c in range(num_labels):
        res = optimize.minimize(lrCostFunction, 
                                initial_theta, 
                                (X, (y == c), Lambda), 
                                jac=True, 
                                method='TNC',
                                options=options)
        all_theta[c]=res.x
    return all_theta

