import scipy.io
import numpy as np
from displayData import displayData
from lrCostFunction import lrCostFunction 
from oneVsAll import oneVsAll

input_layer_size  = 400  
num_labels = 10          

print('Loading and Visualizing Data ...')

dt = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
X = dt['X']
y = dt['y']
m= y.size

# Randomly select 100 data points to display
sel = np.random.permutation(range(100))
sel = sel[0:100]
displayData(X[sel,:])

print('Training One-vs-All Logistic Regression...')

Lambda =0.1
all_theta = oneVsAll(X, y, num_labels, Lambda)


pred = predictOneVsAll(all_theta, X)

accuracy = np.mean(pred==y) * 100
print('\nTraining Set Accuracy: %f\n' % accuracy)
