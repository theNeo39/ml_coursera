import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn
from featureNormalize import featureNormalize
# ================ Part 1: Feature Normalization ================

# Load Data
dt=pd.read_csv('ex1data2.txt',header=None)
X = dt.iloc[:,0:2]
y = dt.iloc[:, 2]
m=y.shape

# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)
print('[mu] [sigma]')
print(mu, sigma)

# Add intercept term to X
X = np.concatenate((np.ones(m).reshape(-1,1), X), axis=1)

# ================ Part 2: Gradient Descent ================
alpha = 0.01
num_iters = 4000

# Init Theta and Run Gradient Descent 
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(J_history, '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: ')
print(theta)
# Estimate the price of a 1650 sq-ft, 3 br house
area=(1650-mu[0])/sigma[0]
broom=(3-mu[1])/sigma[1]
price = np.array([1,area,broom]).dot(theta.T)
print('Predicted price of a 1650 sq-ft, 3 br house')
print(price)

# ================ Part 3: Normal Equations ================

print('Solving with normal equations...')

# Load Data
dt=pd.read_csv('ex1data2.txt',header=None)
X = dt.iloc[:,0:2]
y = dt.iloc[:, 2]
m=y.shape

# Add intercept term to X
X = np.concatenate((np.ones(m).reshape(-1,1),X), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print(theta)
price=0
# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, 1650, 3]).dot(theta)

print("Predicted price of a 1650 sq-ft, 3 br house using normal equations")
print(price)
