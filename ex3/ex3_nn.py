## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from displayData import displayData
from predict import predict

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels   
                          

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scipy.io.loadmat('ex3data1.mat')
X = data['X']
y = data['y']
m=y.size

# Randomly select 100 data points to display
sel = np.random.permutation(range(m))
sel = sel[0:100]

displayData(X[sel,:])


## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weights= scipy.io.loadmat('ex3weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.
pred = predict(Theta1, Theta2, X)
pred=(np.asarray(pred)).reshape(-1,1)

print('Training Set Accuracy: %f\n', np.mean(pred==y)*100)