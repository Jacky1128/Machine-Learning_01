import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

### Target: Use Gaussian model to Implement Anomaly Detection

## STEP_01: Load data set & visualize it
data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex8/ex8/ex8data1.mat')
X = data['X']
print("Shape of X is: ",X.shape)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
ax.set_title("The X in data set ")
plt.show()

## STEP_02: Define a function that receive X，return 2 n-D vectors，mu is the means of each D，sigma2 is variance of each D
# 为了定义概率分布，我们需要两个东西——mean & variance
def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma2 = X.var(axis=0)

    return mu, sigma2       # The process of fitting μ & σ^2

mu, sigma2 = estimate_gaussian(X)
print("The μ(mean) is: ",mu, "; The σ^2(variance) is: ",sigma2)

xplot = np.linspace(0,25,100)
yplot = np.linspace(0,25,100)
Xplot, Yplot = np.meshgrid(xplot,yplot)     # numpy.meshgrid()——create网格点坐标矩阵。
Z = np.exp((-0.5)*((Xplot-mu[0])**2/sigma2[0]+(Yplot-mu[1])**2/sigma2[1]))

fig, ax = plt.subplots(figsize=(12,8))
contour = plt.contour(Xplot, Yplot, Z,[10**-11, 10**-7, 10**-5, 10**-3, 0.1],colors='k')
ax.scatter(X[:,0], X[:,1])
plt.show()

## STEP_03: Select threshold ε by CV set
Xval = data['Xval']     # Cross validation set of X
yval = data['yval']     # Label of different classes (0-Normal; 1-Anomaly）
print("Shape of Xval is: ",Xval.shape, "; Shape of yval is: ",yval.shape)

# Use the tool in SciPy to 计算数据点属于正态分布的概率
from scipy import stats
dist = stats.norm(mu[0], sigma2[0])      #  Get the 概率密度函数 of Mean & variance of the 1st column
print(dist.pdf(X[:,0])[0:50])       # Get the 概率密度 of each point in data set

# 计算并保存给定上述的高斯模型参数的数据集中每个值的概率密度。
p = np.zeros((X.shape[0], X.shape[1]))      # Initialize p as the same scale of X
p[:,0] = stats.norm(mu[0], sigma2[0]).pdf(X[:,0])        # X第一列的概率密度
p[:,1] = stats.norm(mu[1], sigma2[1]).pdf(X[:,1])        # X第二列的概率密度

# We still need to do these operations for CV set (Use the same parameters) in order to get the best threshold ε
pval = np.zeros((Xval.shape[0], Xval.shape[1]))     # Initialize pval as the same scale of Xval
pval[:,0] = stats.norm(mu[0], sigma2[0]).pdf(Xval[:,0])      # Xval第1列的概率密度
pval[:,1] = stats.norm(mu[1], sigma2[1]).pdf(Xval[:,1])      # Xval第2列的概率密度

# Define a function to get the bst threshold ε
# We will calculate the F1-score for different epsilon

def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0      # Initialize

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):     # 等差数列 起始 步长
        preds = pval < epsilon      # 比较结果赋值给preds

        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)     # 逻辑与、类型转换  #真正
        # count the number of 预测1实际1
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)    # 假正
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)    # 假负

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)        # The formular of F1-score

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)
print("The best threshold ε is: ",epsilon, "; Its F1-score is: ",f1)

## STEP_04: Apply the best threshold in data set and visualize the result
# indexes of the values considered to be outliers
outliers = np.where(p < epsilon)
print("The outliers point is: ",outliers)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')
ax.set_title("Anomaly Detection")
plt.show()







