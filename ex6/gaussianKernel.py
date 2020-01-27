import numpy as np


def gaussianKernel(x1, x2, sigma):
    """returns a gaussian kernel between x1 and x2
    and returns the value in sim
    """
    x1=x1.ravel()
    x2=x2.ravel()
    dist=np.sum(np.square(x1-x2))
    sim=np.exp((-1*dist)/(2*np.square(sigma)))
    return sim