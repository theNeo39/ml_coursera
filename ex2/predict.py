import numpy as np
import pandas as pd
from sigmoid import sigmoid

def predict(theta, X):
    p=np.zeros(X.shape[0])
    tx=np.dot(X,theta)
    prob=sigmoid(tx)
    p=np.where((prob>=0.5),1,0)
    return p,prob