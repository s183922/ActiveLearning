import pickle 
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as st
from itertools  import  combinations
import pandas as pd

path = sys.path[0]
os.chdir(path)
filein= open("QBCTest.obj",'rb')
obj1 = pickle.load(filein)

filein= open("UncertaintyTest.obj",'rb')
obj2 = pickle.load(filein)

filein = open("ExpectedGradient.obj", 'rb')
obj3 = pickle.load(filein)
obj = [obj3] + [obj1] + obj2



fig = plt.figure(figsize = (10,20))
fig.add_subplot(1,3,1)
methods = ["EMC", "QBC","Least Conf.", "Entr.", "Marg.", "Basel."]
cols = ["C0","C1","C2","C3","C4", "C5"]
for i in range(len(methods)):
    L = []
    for j in range(20):
        L.append([*zip(*obj[i][j])][1])
    mu = np.mean(L, axis = 0)
    plt.plot(np.arange(100,200),mu)
plt.legend(methods)
plt.grid(linestyle = "dashed")
plt.xlabel("Number of observations in training data")
plt.ylabel("Accuracy Score")
plt.title("Results of Active Learning Methods compared to baseline")


fig.add_subplot(1,3,2)
for i in range(len(methods)):
    L = []
    for j in range(20):
        L.append([*zip(*obj[i][j])][1][-1])
    plt.boxplot(L, positions= [i], widths= 0.70,
                boxprops=dict(facecolor = cols[i], color = cols[i], alpha = 0.7),
                whiskerprops=dict(color = cols[i]),
                medianprops = dict(color = 'black'),
                flierprops = dict(color = cols[i]),
                capprops= dict(color = cols[i]),
                patch_artist = True)
plt.grid(axis = "y", linestyle = "dashed")
plt.xticks(np.arange(6), methods)
plt.ylabel("Accuracy Score")
plt.title("Accuracy score with 200 Datapoints")
fig.add_subplot(1,3,3)
alpha = 0.05
G = []
for i in range(len(methods)):
    L = []
    for j in range(20):
        L.append([*zip(*obj[i][j])][1][-1])
    plt.scatter([i], np.mean(L))
    conf_int = st.t.interval(0.95, len(L)-1, loc=np.mean(L), scale=st.sem(L))
    plt.hlines(conf_int, i-0.3, i+0.3, colors = cols[i])
    plt.vlines(i, conf_int[0], conf_int[1], colors = cols[i], linestyles= "dashed")

    G.append(L)
plt.xticks(np.arange(6), methods)
plt.grid(axis = "y", linestyle="dashed")
plt.ylabel("Accuracy Score")
plt.title("Confidence interval of mean accuracy score with 200 Datapoints")
plt.show()

comb = list(combinations(np.arange(len(methods)), 2))
d = {}
bonferroni = 0.05/len(comb)
for i in comb:
    Name = "{:} VS {:}".format(methods[i[0]], methods[i[1]])
    t_obs = st.ttest_ind(G[i[0]], G[i[1]])[0]
    p_val = st.ttest_ind(G[i[0]], G[i[1]])[1]
    if p_val < bonferroni:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = ""
    d.update({Name: [t_obs, p_val, sig]})

df = pd.DataFrame.from_dict(d)
df.index = ["Test Statistic", "p-value", "signifcance"]
print("Adjusted critical value (Bonferroni): {:}".format(bonferroni))
print("Significance * = <0.05")
print("Significance ** = < 0.005")
print(df.T.to_latex())

