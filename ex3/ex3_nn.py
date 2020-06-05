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
                          


print('Loading and Visualizing Data ...')

data = scipy.io.loadmat('ex3data1.mat')
X = data['X']
y = data['y']
m=y.size

# Randomly select 100 data points to display
sel = np.random.permutation(range(m))
sel = sel[0:100]

displayData(X[sel,:])


print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weights= scipy.io.loadmat('ex3weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

pred = predict(Theta1, Theta2, X)
pred=(np.asarray(pred)).reshape(-1,1)

print('Training Set Accuracy: %f\n', np.mean(pred==y)*100)