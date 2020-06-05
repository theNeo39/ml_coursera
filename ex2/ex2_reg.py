import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import seaborn as sns
#from ml import mapFeature, plotData, plotDecisionBoundary
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg
from sigmoid import sigmoid
from predict import predict

# Initialization
dt =pd.read_csv('ex2data2.txt', header=None)
X=dt.iloc[:,0:2]
y=dt.iloc[:,2]
y1=X.loc[(y==1),:]
y0=X.loc[(y==0),:]
sns.scatterplot(x=y1.loc[:,0],y=y1.loc[:,1],marker='+')
sns.scatterplot(x=y0.loc[:,0],y=y0.loc[:,1],marker='o')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# =========== Part 1: Regularized Logistic Regression ============
X=np.asarray(X)
X=mapFeature(X[:, 0], X[:, 1])
# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
Lambda =1
# Compute and display initial cost and gradient for regularized logistic
# regression
cost,grad = costFunctionReg(initial_theta, X, y, Lambda)


print('Cost at initial theta (zeros): %f' % cost)
# ============= Part 2: Regularization and Accuracies =============

# Optimize and plot boundary
Lambda = 1
res = optimize.minimize(costFunctionReg, initial_theta, method='TNC',jac=True, args=(X, y,Lambda), options={'maxiter': 1000})

theta = res.x
cost = res.fun

tx=X[:,1]
ty=X[:,2]
z = np.zeros((tx.size, tx.size))
        # Evaluate z = theta*x over the grid
for i, ui in enumerate(tx):
   for j, vj in enumerate(ty):
       z[i, j] =np.dot(mapFeature(ui, vj), theta)

z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
plt.contour(tx, ty, z,levels=[-0.005],linewidths=2, colors='g')

# Print to screen
print('lambda = ' + str(Lambda))
print('Cost at theta found by scipy: %f' % cost)
print('theta:', ["%0.4f" % i for i in theta])

# Compute accuracy on our training set
preds,probs = predict(theta, X)
print(np.mean(preds==y)*100)