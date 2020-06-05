import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warmUpExercise import warmUpExercise
from computeCost import computeCost
from gradientDescent import gradientDescent
from mpl_toolkits.mplot3d import Axes3D


print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')
warmup = warmUpExercise()

data=pd.read_csv('ex1data1.txt',header=None)
X=data.iloc[:,0:1]
y=data.iloc[:,1]

sns.scatterplot(X[0],y)
plt.xlabel('population in cities')
plt.ylabel('profits in cities')

print('Running Gradient Descent ...')
m=y.shape[0]
X=np.concatenate((np.ones((m,1)),X),axis=1)
theta = np.zeros(2)
J = computeCost(X, y, theta)
print('cost: %0.4f '% J)


iterations = 1500
alpha = 0.01

# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
# print theta to screen
print('Theta found by gradient descent: ')
print('%s %s \n' % (theta[0], theta[1]))

# Plot the linear fit
plt.figure()
sns.scatterplot(X[:,1],y)
sns.lineplot(X[:,1],np.dot(X,theta), label='Linear regression')

# Predict values for population sizes of 35,000 and 70,000
predict1 = theta.T.dot(np.array([1, 3.5]))
predict2 = theta.T.dot(np.array([1, 7]))
print('For population = 35,000, we predict a profit of %.4f'% (predict1*10000))
print('For population = 70,000, we predict a profit of %.4f'% (predict2*10000))


# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(X, y, np.asarray([theta0, theta1]))
        
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