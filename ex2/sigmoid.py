import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""
    z=-1*z
    g=(1.0/(1.0+np.exp(z)))
    return g