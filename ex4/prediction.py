import sys
sys.path.insert(1, 'D:\GitProjects\Machine Learning\ML Coursera\ex2')
import numpy as np
from sigmoid import sigmoid

def prediction(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape
    
    z2= np.concatenate([np.ones((m, 1)), X], axis=1)
    z2=np.dot(Theta1,z2.T).T
    a2=sigmoid(z2)
    z3=np.concatenate([np.ones((m, 1)), a2], axis=1)
    z3=np.dot(Theta2,z3.T).T
    a3=sigmoid(z3)
    p=np.argmax(a3,axis=1)

    return p

