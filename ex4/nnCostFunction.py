import sys
sys.path.insert(1, 'D:\GitProjects\Machine Learning\ML Coursera\ex2')
import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

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
    grad=0
    
    #forward Propagation
    
    layer1=np.concatenate((np.ones(m).reshape(-1,1), X), axis=1)
    z2=(np.dot(Theta1,layer1.T)).T
    z2=sigmoid(z2)
    layer2=np.concatenate((np.ones(m).reshape(-1,1),z2),axis=1)
    z3=(np.dot(Theta2,layer2.T)).T
    z3=sigmoid(z3)
    temp_J=np.zeros(m)
    for i in range(num_labels):
        y_vector=y.copy()
        y_vector=np.where(y_vector==i,1,0)
        t1=(-1*y_vector)*(np.log(z3[:,i]))
        t2=(1-y_vector)*(np.log(1-z3[:,i]))
        temp_J=temp_J+np.subtract(t1,t2)
    J=(np.sum(temp_J))/m
    
    #forward Propagation with regularization
    t1sum=np.sum(np.square(Theta1[:,1:]))
    t2sum=np.sum(np.square(Theta2[:,1:]))
    J_reg=(Lambda*(t1sum+t2sum))/(2*m)
    J=J+J_reg
    
    
    

#
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a 
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the 
#               first time.
#
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#



    # -------------------------------------------------------------

    # =========================================================================

    # Unroll gradient
    #grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))


    return J, grad