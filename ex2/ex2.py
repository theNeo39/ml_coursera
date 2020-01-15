import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
from sigmoid import sigmoid
from costFunction import costFunction
from gradientFunction import gradientFunction
from ml import plotData, plotDecisionBoundary
from predict import predict

## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.py done
#     costFunction.py done
#     gradientFunction.py done
#     predict.py
#     costFunctionReg.py
#     gradientFunctionReg.py
#     n.b. This files differ in number from the Octave version of ex2.
#          This is due to the scipy optimization taking only scalar
#          functions where fmiunc in Octave takes functions returning
#          multiple values.
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

#from ml import plotData, plotDecisionBoundary
# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

dt=pd.read_csv('ex2data1.txt',header=None)
X=dt.iloc[:,0:2]
y=dt.iloc[:,2]
#sns.scatterplot(x=X[0],y=X[1],hue=y)

# # ============ Part 2: Compute Cost and Gradient ============
# #  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape
# Add intercept term to x and X_test
X = np.concatenate((np.ones(m).reshape(-1,1), X), axis=1)
# Initialize fitting parameters
initial_theta =np.zeros(n+1)
initial_theta=(np.asarray([-24,0.2,0.2])).T
temp=(np.dot(initial_theta.T,X.T)).T
g=sigmoid(temp)

# Compute and display initial cost and gradient
cost,grad = costFunction(initial_theta, X, y)
print('Cost at initial theta (zeros): %f' % cost)
print('Gradient at initial theta (zeros): ',grad)

# ============= Part 3: Optimizing using scipy  =============
res = optimize.minimize(costFunction, initial_theta, method='TNC',jac=True, args=(X, y), options={'maxiter': 1000})

theta = res.x
cost1 = res.fun

# Print theta to screen
print('Cost at theta found by scipy: %f' % cost1)
print('theta:',theta)
# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Labels and Legend
plt.legend(['Admitted', 'Not admitted'], loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

#  ============== Part 4: Predict and Accuracies ==============

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2
X1=np.array([1,45,85])
pred,prob = predict(theta,X1)
print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)
# Compute accuracy on our training set
preds,probs = predict(theta, X)
print(np.mean(preds==y)*100)