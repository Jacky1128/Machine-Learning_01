import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat        # To load the data set with matlab's form(.m)
from sklearn.metrics import classification_report   # This package is classification report

# Target: 这个部分需要你实现手写数字（0到9）的识别。你需要扩展之前的逻辑回归，并将其应用于Multi-class Classification

## STEP_01: Get data set & achieve its visualization
# 这是一个MATLAB格式的.m文件，其中包含5000个20*20像素的手写字体图像，以及他对应的数字。
# 另外，数字0的y值，对应的是10 用Python读取我们需要使用SciPy

# Get data
data = loadmat('C:/Users/JackyWang28/Desktop/ex3data1.mat')
print(data)
print("The shape of X is ",data['X'].shape, "The shape of y is ",data['y'].shape)

# Show 100 data randomly
sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
sample_images = data['X'][sample_idx, :]
print("The samples selected are: ",sample_images)       # Show the samples that been selected

# Data visualization
fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T,cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()

## STEP_02: Vectorize the Logistic Regression
# Target: Use Multi-class Logistic Regression to make a classifier
# There are 3 types of numbers here. so we will train 10 different classifers
# Vectorize the Logistic Regression to avoid loop

# Define Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define Cost function
def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

# Define Gradient
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()