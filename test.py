import pickle 
import os, sys
import matplotlib.pyplot as plt
import numpy as np
path = sys.path[0]
os.chdir(path)

filein= open("UncertaintyTest.obj",'rb')
obj = pickle.load(filein)
methods = ["Least Confident", "Entropy", "Margin", "Baseline"]
for i in range(4):
    L = []
    for j in range(20):
        L.append([*zip(*obj[i][j])][1])
    mu = np.mean(L, axis = 0)
    plt.plot(np.arange(100,200),mu)
plt.legend(methods)
plt.grid()
plt.show()
