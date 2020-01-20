import sys
sys.path.insert(1, 'D:\GitProjects\Machine Learning\ML Coursera\ex2')
import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda=0.0):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1)).copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1))).copy()

# Setup some useful variables
    m = X.shape[0]
    derv3=np.zeros((m,num_labels))
    
    
    #forward Propagation
    
    l1p1=np.concatenate((np.ones(m).reshape(-1,1), X), axis=1)
    z2=(np.dot(Theta1,l1p1.T)).T
    a2=sigmoid(z2)
    l2p1=np.concatenate((np.ones(m).reshape(-1,1),a2),axis=1)
    z3=(np.dot(Theta2,l2p1.T)).T
    a3=sigmoid(z3)
    temp_J=np.zeros(m)
    for i in range(num_labels):
        y_vector=y.copy()
        y_vector=np.where(y_vector==i,1,0)
        t1=(-1*y_vector)*(np.log(a3[:,i]))
        t2=(1-y_vector)*(np.log(1-a3[:,i]))
        temp_J=temp_J+np.subtract(t1,t2)
        derv3[:,i]=np.subtract(a3[:,i],y_vector)
    J=(np.sum(temp_J))/m
    #forward Propagation with regularization
    t1sum=np.sum(np.square(Theta1[:,1:]))
    t2sum=np.sum(np.square(Theta2[:,1:]))
    J_reg=(Lambda*(t1sum+t2sum))/(2*m)
    J=J+J_reg

    
    #backpropagation
    derv2=((np.dot(Theta2[:,1:].T,derv3.T))*((a2*(1-a2)).T)).T
    D1=np.dot(derv2.T,l1p1)
    D2=np.dot(derv3.T,l2p1)
    Theta1_grad=D1/m
    Theta2_grad=D2/m
    Theta1_grad[:,1:]=Theta1_grad[:,1:] +(Lambda/m)*Theta1[:,1:]
    Theta2_grad[:,1:]=Theta2_grad[:,1:] +(Lambda/m)*Theta2[:,1:]
    
    
    
    


    # -------------------------------------------------------------

    # =========================================================================

    # Unroll gradient
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    return J,grad