import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat        # To load the data set with matlab's form(.m)
from sklearn.metrics import classification_report   # This package is classification report

# Target: Achieve the fucntion of predicting handwritting numbers by Forward Propagation Neural Networks
# Neural Networks will return the largest (hθ(x))k as result

## STEP_01: Load the data set of the weights(parameters) of Neural Networks
# Get data
data = loadmat('C:/Users/JackyWang28/Desktop/ex3data1.mat')
print(data)
print("The shape of X is ",data['X'].shape, "The shape of y is ",data['y'].shape)

rows = data['X'].shape[0]
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

# Load weights
weight = loadmat("C:/Users/JackyWang28/Desktop/ex3weights.mat")
theta1, theta2 = weight['Theta1'], weight['Theta2']
print("The shape of theta1 is: ",theta1.shape, "; The shape of theta2 is: ",theta2.shape)

## STEP_02: Dealing with the raw data, attach it to Neural Networks tradition
# Insert the constant term(x0) in X and change data type into matrix
X2 = np.matrix(np.insert(data['X'], 0, values=np.ones(X.shape[0]), axis=1))     # Add cols of x0 to raw data [bias]
y2 = np.matrix(data['y'])
print("The shape of X2 is: ",X2.shape, "; The shape of y2 is: ",y2.shape)

# Define Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Name all the variables with Neural Networks rules
a1 = X2     # a1 means activation1
z2 = a1 * theta1.T     # a2 = g(z2)
print("The shape of z2 is: ",z2.shape)      # Check the shape of z2

a2 = sigmoid(z2)    # a2 means activation2; a2 = g(z2), g(x) = sigmoid(z)
print("The shape of a2 is: ",a2.shape)      # Check the shape of a2

a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)      # Add cols of a(2)0 to raw data [bias]
z3 = a2 * theta2.T      # z(3) = a(2)*θ(2)
print("The shape of z3 is: ",z3.shape)      # Check the shape of z3

a3 = sigmoid(z3)    # a3 means activation3; a3 = g(z3)
print("The activation in the 3rd layer is: ",a3)


y_pred2 = np.argmax(a3, axis=1) + 1
print("The shape of y_pred2 is: ",y_pred2.shape)

print(classification_report(y2, y_pred2))

# 后三行不理解
# 虽然人工神经网络是非常强大的模型，但训练数据的准确性并不能完美预测实际数据，在这里很容易过拟合。