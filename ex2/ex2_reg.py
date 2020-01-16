import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
from ml import mapFeature, plotData, plotDecisionBoundary
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg
from sigmoid import sigmoid
from predict import predict

# Plot Boundary
def plotBoundary(theta, X, y):
    plotDecisionBoundary(theta, X, y)
    plt.title(r'$\lambda$ = ' + str(Lambda))

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')


# Initialization

# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

dt =pd.read_csv('ex2data2.txt', header=None)
X=dt.iloc[:,0:2]
y=dt.iloc[:,2]
plotData(X.values, y.values)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# =========== Part 1: Regularized Logistic Regression ============

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X=np.asarray(X)
X=mapFeature(X[:, 0], X[:, 1])
# Initialize fitting parameters
initial_theta = np.ones(X.shape[1])

# Set regularization parameter lambda to 1
Lambda =10
# Compute and display initial cost and gradient for regularized logistic
# regression
cost,grad = costFunctionReg(initial_theta, X, y, Lambda)


print('Cost at initial theta (zeros): %f' % cost)
# ============= Part 2: Regularization and Accuracies =============

# Optimize and plot boundary
Lambda = 1.0
res = optimize.minimize(costFunctionReg, initial_theta, method='TNC',jac=True, args=(X, y,Lambda), options={'maxiter': 1000})
#result = Optimize(Lambda)
theta = res.x
cost = res.fun

# Print to screen
print('lambda = ' + str(Lambda))
print('Cost at theta found by scipy: %f' % cost)
print('theta:', ["%0.4f" % i for i in theta])

# Compute accuracy on our training set
preds,probs = predict(theta, X)
print(np.mean(preds==y)*100)