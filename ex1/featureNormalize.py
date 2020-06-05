import numpy as np


def featureNormalize(X):
    X_norm=X.copy()
    mu=np.zeros(X_norm.shape[1])
    sigma=np.zeros(X_norm.shape[1])
    for i in range(X_norm.shape[1]):
        mu[i]=np.mean(X_norm[i])
        sigma[i]=np.std(X_norm[i])
        X_norm[i]=(X_norm[i]-mu[i])/sigma[i]
    return X_norm, mu, sigma
