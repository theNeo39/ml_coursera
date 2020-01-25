import scipy.io
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve

## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m

## =========== Part 1: Loading and Visualizing Data =============
print('Loading and Visualizing Data ...')
dt = scipy.io.loadmat('ex5data1.mat')
X = dt['X'].squeeze()
y = dt['y'].squeeze()
Xval = dt['Xval'].squeeze()
yval = dt['yval'].squeeze()
Xtest = dt['Xtest'].squeeze()
ytest=dt['ytest'].squeeze()
m = X.shape[0]
sns.scatterplot(X,y)
plt.ylabel('Water flowing out of the dam')            
plt.xlabel('Change in water level')

## =========== Part 2: Regularized Linear Regression Cost =============
X=np.concatenate((np.ones(m).reshape(-1,1),X.reshape(-1,1)),axis=1)
theta = np.array([1, 1])
J = linearRegCostFunction(theta,X,y,1)[0]
print('Cost at theta = [1  1]: %f \n(this value should be about 303.993192)\n' % J)

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#
theta = np.array([1, 1])
J, grad = linearRegCostFunction(theta,X,y,1)

print('Gradient at theta = [1  1]:  [%f %f] \n(this value should be about [-15.303016 598.250744])\n' %(grad[0], grad[1]))
## =========== Part 4: Train Linear Regression =============
Lambda = 0
theta = trainLinearReg(X,y,Lambda)

#  Plot fit over the data
plt.plot(X[:,1],np.dot(X,theta), '-')
plt.close()
## =========== Part 5: Learning Curve for Linear Regression =============
Lambda = 0
error_train, error_val = learningCurve(X, y,np.column_stack((np.ones(Xval.shape[0]), Xval)), yval, Lambda)

plt.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=3)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])

print('Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X[:,1], p)

X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.column_stack((np.ones(m), X_poly))                   

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_test))        # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_val))           # Add Ones

print('Normalized Training Example 1:')
print(X_poly[0, :])

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of Lambda. The code below runs polynomial regression with 
#  Lambda = 0. You should try running the code with different values of
#  Lambda to see how the fit and learning curve change.
#
Lambda = 0
theta = trainLinearReg(X_poly, y, Lambda,maxiter=55)

# Plot training data and fit
plt.figure(2)
plt.plot(X[:,1], y, 'ro', ms=10, mew=1.5, mec='k')
plotFit(min(X[:,1]), max(X[:,1]), mu, sigma, theta, p)

plt.xlabel('Change in water level (x)')            # Set the y-axis label
plt.ylabel('Water flowing out of the dam (y)')     # Set the x-axis label
plt.title('Polynomial Regression Fit (Lambda = %f)' % Lambda)


error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, Lambda)
plt.figure(3)
plt.plot(range(m), error_train, label='Train')
plt.plot(range(m), error_val, label='Cross Validation')
plt.title('Polynomial Regression Learning Curve (Lambda = %f)' % Lambda)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.legend()
print('Polynomial Regression (Lambda = %f)\n\n' % Lambda)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

## =========== Part 8: Validation for Selecting Lambda =============

Lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plt.figure(4)
plt.plot(Lambda_vec, error_train, '-o', Lambda_vec, error_val,'-o',lw=2)
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.legend(['Train', 'Cross Validation'])
print('Lambda\t\tTrain Error\tValidation Error')
for i in range(Lambda_vec.size):
    print(' %f\t%f\t%f' % (Lambda_vec[i], error_train[i], error_val[i]))
