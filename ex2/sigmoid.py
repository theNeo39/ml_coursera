import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""
    z=-1*z
    g=(1/(1+np.exp(z)))
    return g