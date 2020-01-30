import numpy as np
def projectData(X, U, K):
    """computes the projection of
    the normalized inputs X into the reduced dimensional space spanned by
    the first K columns of U. It returns the projected examples in Z.
    """

    Ureduce=U[:,:K]
    Z=np.dot(X,Ureduce)


    return Z
