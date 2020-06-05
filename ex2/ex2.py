import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
from sigmoid import sigmoid
from costFunction import costFunction
from gradientFunction import gradientFunction
from predict import predict


dt=pd.read_csv('ex2data1.txt',header=None)
X=dt.iloc[:,0:2]
y=dt.iloc[:,2]

sns.scatterplot(x=X[0],y=X[1],hue=y)

m= X.shape[0]
# Add intercept term to x and X_test
X=np.concatenate((np.ones((m,1)), X), axis=1)
# Initialize fitting parameters
initial_theta =np.zeros(X.shape[1])
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
plt.figure()
y1=X[(y==1),1:]
y0=X[(y==0),1:]
sns.scatterplot(x=y1[:,0],y=y1[:,1],marker='+')
sns.scatterplot(x=y0[:,0],y=y0[:,1],marker='o')
fp,fpred=predict(theta,X)
ty=-1*(theta[0]*X[:,0]+theta[1]*X[:,1])/theta[2]
sns.lineplot(X[:,1],ty)
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