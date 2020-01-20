from sigmoid import sigmoid
import numpy as np

def sigmoidGradient(z):
    """computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element."""
    g=sigmoid(z)*(1-sigmoid(z))
    return g
