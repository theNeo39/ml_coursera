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
      
        # Set options for minimize
    options = {'maxiter': 10}
    
        # Run minimize to obtain the optimal theta. This function will 
        # return a class object where theta is in `res.x` and cost in `res.fun`
    for c in range(num_labels):
        res = optimize.minimize(lrCostFunction, 
                                initial_theta, 
                                (X, (y == c), Lambda), 
                                jac=True, 
                                method='TNC',
                                options=options)
        print('for c=0 theta is') 
        print(res.x)
        all_theta[c]=res.x
    
# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda. 
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

    # Set Initial theta

    # This function will return theta and the cost



# =========================================================================

    return all_theta

