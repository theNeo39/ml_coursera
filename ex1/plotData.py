import matplotlib.pyplot as plt
import numpy as np

def plotData(data):
    plt.scatter(x=data[0],y=data[1])
    plt.xlabel('Population in 10,000\'s')
    plt.ylabel('Profit in $10,000')
    plt.show()
