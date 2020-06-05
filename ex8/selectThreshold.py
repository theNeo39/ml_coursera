import numpy as np
import math

def selectThreshold(yval, pval):
    """
    finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    bestEpsilon = 0
    bestF1 = 0

    for epsilon in np.linspace(1.01*min(pval), max(pval), 1000):
        
        tp_val=np.where(pval < epsilon,1,0)
        tp=np.sum((tp_val==1)& (yval==1))
        tn=np.sum((tp_val==0)& (yval==0))
        fp=np.sum((tp_val==1)& (yval==0))
        fn=np.sum((tp_val==0)& (yval==1))
        prec=tp/(tp+fp)
        rec=tp/(tp+fn)
        F1=2*prec*rec/(prec+rec)
        if F1 > bestF1:
           bestF1 = F1
           bestEpsilon = epsilon

    return bestEpsilon, bestF1






