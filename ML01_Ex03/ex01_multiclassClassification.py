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
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))       # 1st step in fomular
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))     # 2nd step in fomular
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))        # Fomular in shimo file
    return np.sum(first - second) / len(X) + reg        # Finish the calculation process of regularized Cost function

# Define Regularized Gradient
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)        # Change the data type into matrix

    parameters = int(theta.ravel().shape[1])        # shape[1] is the width of matrix
    error = sigmoid(X * theta.T) - y        # a element(part) of the θ updating fomular: [hθ(xi)-yi]

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)       # Well updated θj

    # Intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


## STEP_03: Build Classifier based on "One vs All" strategy
# In this task, we own 10 possible types，the guidline is k types(labels) need k classifiers
# Each classifier does the same thing as normal classifier
# We put the whole classifier training process in a function to calculate final weights in each classifier of 10
# Then return then weights as a k * (n + 1) arrary，n is the number of parameters(weights)

from scipy.optimize import minimize     # To calculate θ_min

# Define Multi-class classifier
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]       # The height(num of rows)
    params = X.shape[1]     # The width of X

    # k * (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)  # np.ones() generate an array filled with 1

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x

    return all_theta
# NOTICE: The most challenging point of vetorized coding is dealing with all the matrix correctly(especially the dimention)

rows = data['X'].shape[0]
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

print("Shape of X is: ",X.shape, "Shape of y_0 is: ",y_0.shape, "Shape of theta is: ",theta.shape)
print("Shape of all_theta is: ",all_theta.shape)
# NOTICE: θ is a 1-dim arrary, so when we change it to (1×401) matrix to adapt the gradient calculation process，
print("The label nums in y is: ",np.unique(data['y']))     # Check how many labels in y

## STEP_04: Calculate all the theta_min and predict the results
# Use one_vs_all function to calculate all_theta_min (set learning rate by us)
all_theta = one_vs_all(data['X'], data['y'], 10, 1)
print(all_theta)

# Define the predict_all function
# We can use predict_all function to predict each example and see the operation process of classifier
def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax

y_pred = predict_all(data['X'], all_theta)
print(classification_report(data['y'], y_pred))

