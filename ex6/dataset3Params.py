import numpy as np
from sklearn import svm


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.
    C = [0.01,0.03,0.1,0.3,1,3,10,30]
    gamma=[0.1,0.3,1,3,10,30,100,300]
    pred=np.zeros((8,8))
    for i in range(len(C)):
        for j in range(len(gamma)):
            clf=svm.SVC(C=C[i],kernel='rbf',gamma=gamma[j])
            model=clf.fit(X,y)
            pred_val=model.predict(Xval)
            pred[i][j]=np.mean(pred_val!=yval)
    index=np.argmin(pred)
    C_i=(index/len(C)).astype(int)
    gamma_i=index%len(gamma)
    return C[C_i],gamma[gamma_i]
