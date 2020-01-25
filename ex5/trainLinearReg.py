from scipy import optimize
import numpy as np
from linearRegCostFunction import linearRegCostFunction

def trainLinearReg(X, y,Lambda=0.0, maxiter=1000):
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])
    options = {'maxiter': maxiter}

    # Minimize using scipy
    res = optimize.minimize(linearRegCostFunction, initial_theta, args=(X,y,Lambda),jac=True, method='TNC', options=options)
    return res.x