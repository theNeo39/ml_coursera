import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following modules
#  in this exericse:
#
#     warmUpExercise.py done
#     plotData.py done
#     gradientDescent.py done
#     computeCost.py done
#     gradientDescentMulti.py done
#     computeCostMulti.py done
#     featureNormalize.py done
#     normalEqn.py 

# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')
warmup = warmUpExercise()

# ======================= Part 2: Plotting =======================
data=pd.read_csv('ex1data1.txt',header=None)
X=data.iloc[:,0:1]
y=data.iloc[:,1]

# Plot Data
# Note: You have to complete the code in plotData.py
print('Plotting Data ...')
plotData(data)

# =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...')
m=y.shape
X=np.concatenate((np.ones(m).reshape(-1,1),X),axis=1)
theta = np.zeros(2)
J = computeCost(X, y, theta)
print('cost: %0.4f '% J)

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
# print theta to screen
print('Theta found by gradient descent: ')
print('%s %s \n' % (theta[0], theta[1]))

# Plot the linear fit
plotData(data)
plt.plot(X[:,1].T,(np.dot(theta.T,X.T)), '-', label='Linear regression')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta.T)
predict2 = np.array([1, 7]).dot(theta.T)
print('For population = 35,000, we predict a profit of %.4f'% (predict1*10000))
print('For population = 70,000, we predict a profit of %.4f'% (predict2*10000))

# ============= Part 4: Visualizing J(theta_0, theta_1) =============

# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(X, y, np.asarray([theta0, theta1]).T)
        
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# surface plot
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')

# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = plt.subplot(122)
plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
plt.title('Contour, showing minimum')

# =============Use Scikit-learn =============
regr = linear_model.LinearRegression(fit_intercept=False, normalize=True)
regr.fit(X, y)

print('Theta found by scikit: ')
print('%s %s \n' % (regr.coef_[0], regr.coef_[1]))

predict1 = np.array([1, 3.5]).dot(regr.coef_)
predict2 = np.array([1, 7]).dot(regr.coef_)
print('For population = 35,000, we predict a profit of %.4f'% (predict1*10000))
print('For population = 70,000, we predict a profit of %.4f'% (predict2*10000))

#plotData(data)
#plt.plot(X[:, 1],  X.dot(regr.coef_.T), '-', color='black', label='Linear regression wit scikit')
