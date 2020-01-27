## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
import numpy as np
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from visualizeBoundaryLinear import visualizeBoundaryLinear
from gaussianKernel import gaussianKernel
from visualizeBoundary import visualizeBoundary
from dataset3Params import dataset3Params


## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data ...')

# Load from ex6data1: 
# You will have X, y in your environment
dt = scipy.io.loadmat('ex6data1.mat')
X = dt['X']
y = dt['y'].squeeze()

# Plot training data
sns.scatterplot(x=X[:,0],y=X[:,1],hue=y)

## ==================== Part 2: Training Linear SVM ====================
print('Training Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)

C = 1 #100
clf = svm.SVC(C=C, kernel='linear')
model = clf.fit(X, y)
co=model.coef_
inter=model.intercept_
visualizeBoundaryLinear(X, y, model)

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('Evaluating the Gaussian Kernel ...')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)
print('Gaussian Kernel between x1 = [1 2 1], x2 = [0 4 -1], sigma = %0.5f : ' \
       '\t%f\n(this value should be about 0.324652)\n' % (sigma, sim))

## =============== Part 4: Visualizing Dataset 2 ================

print('Loading and Visualizing Data ...')

# Load from ex6data2:
dt = scipy.io.loadmat('ex6data2.mat')
X = dt['X']
y = dt['y'].flatten()

# Plot training data
plt.figure()
sns.scatterplot(X[:,0],X[:,1],hue=y)

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the
#  SVM classifier.
#
print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

# SVM Parameters
C = 1
clf = svm.SVC(C=C, kernel='rbf', gamma=100)
model = clf.fit(X, y)
visualizeBoundary(X, y, model)

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#

print('Loading and Visualizing Data ...')

# Load from ex6data3:
# You will have X, y in your environment
dt = scipy.io.loadmat('ex6data3.mat')
X = dt['X']
y = dt['y'].flatten()
Xval = dt['Xval']
yval = dt['yval'].flatten()

# Plot training data
plt.figure()
sns.scatterplot(X[:,0],X[:,1],hue=y)


## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

# Try different SVM Parameters here
C,gamma = dataset3Params(X, y, Xval, yval)
# Train the SVM

clf = svm.SVC(C=C, kernel='rbf',gamma=gamma)
model = clf.fit(X, y)
visualizeBoundary(X, y, model)