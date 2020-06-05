import matplotlib.pyplot as plt
import numpy as np
from math import isinf
from multivariateGaussian import multivariateGaussian


def visualizeFit(X, mu, sigma2):
    """
    This visualization shows you the
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariateGaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1],'bx')
    # Do not plot if there are infinities
    if not isinf(np.sum(Z)):
        plt.contour(X1[0], X1[1], Z, 10.0**np.arange(-20, 0, 3))
