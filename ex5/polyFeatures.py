import numpy as np

def polyFeatures(X, p):
    """takes a data matrix X (size m x 1) and
    maps each example into its polynomial features where
    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
    """
# You need to return the following variables correctly.
    X_poly = np.zeros((X.shape[0],p))
    for i in range(1,(p+1)):
        if i==1:
            X_poly=X.reshape(-1,1)
        else:
            X_poly=np.concatenate((X_poly,np.power(X.reshape(-1,1),i)),axis=1)
    return X_poly
