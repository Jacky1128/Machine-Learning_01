import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder     # Use OneHotEncoder to encode y label

np.seterr(divide='ignore',invalid='ignore')

# Target: We will deal with the handwritting data set, then operate the whole Neural Network process(weight has given)
# automatically learn parameters θ through backpropagation Neural Networks
# data set: 5000 pages of 20*20像素's handwritting digital data set, 对应的数字（1-9，0对应10）

## STEP_01: Load, deal with the data set and visualize it
# Load X and y from data set
data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex4/ex4/ex4data1.mat')
print("The raw data set is: ",data)

X = data['X']
y = data['y']
print("View the shape of the X and y: ")
print("Shape of X is: ",X.shape, " Shape of y is: ",y.shape)

# Load initial weight from data set
weight = loadmat("C:/Users/JackyWang28/Desktop/machine-learning-ex4/ex4/ex4weights.mat")
theta1, theta2 = weight['Theta1'], weight['Theta2']
print("Shape of theta1 is: ",theta1.shape, " Shape of theta2 is: ",theta2.shape)

# Visulize the data
sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
sample_images = data['X'][sample_idx, :]
fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T,cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()

## STEP_02: Achieve适用于任何数据集、包括任意数量的输入输出单元神经网络的 Cost Function & Gradient Function
## Without Regularization

# Define Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define Forward Propagation Function
def forward_propagate(X, theta1, theta2):      #
    m = X.shape[0]      # Get the height of X

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)     # Add a column filled with 1 as the 1st column of X (bias unit)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)       # Calculate a2 and add bias unit at the same time
    z3 = a2 * theta2.T
    h = sigmoid(z3)     # The output layer

    return a1, z2, a2, z3, h

# Define the Cost Function without Regularization
def cost(theta1, theta2, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]      # Get the height of X
    X = np.matrix(X)
    y = np.matrix(y)        # Change into matrix type

    # run the Forward Propagation to get a1, z2, z3, and hypothesis function h;
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # Apply all these variables got above to compute the Cost Function J(θ)
    J = 0       # Initialize J(θ)
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)       # The signal '+=' indicates the second sum

    J = J / m
    return J

# Deal with(encode) y label
# The y lable in raw data set is a 5000*1 vector, we've to make it a 5000*10 matrix,
# The reason why make 1 to 10 is to mark y in a form like: from y = 2 to y = [0 1 0 ... 0]
encoder = OneHotEncoder(sparse=False)
# 默认sparse = True,return a 稀疏矩阵的对象,一般要调用toarray()方法转化成array对象。
# if let sparse = False，则直接生成array对象，可直接使用。
y_onehot = encoder.fit_transform(y)
print("Shape of y_onehot is: ",y_onehot.shape)
print("y[0] is: ",y[0], " y_onehot[0,:] is: ",y_onehot[0,:])      # y0 is a number 0;

# Artificially make Initialization
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# Use functions set above to calculate the Cost Function(without Regularization)
print("The Cost is: ",cost(theta1, theta2, input_size, hidden_size, num_labels, X, y_onehot, learning_rate))

## STEP_03: Regularize the Cost Function
# The function are appropriate to every theta
# Define a function to calculate Regularized Cost Function
def costReg(theta1, theta2, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # run the Forward Propagation to get a1, z2, z3, and hypothesis function h;
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # Apply all these variables got above to compute the Cost Function J(θ)
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m
    # the code above are all the same as function:cost()

    # add the cost regularization term [the only different part compared with cost()]
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    return J

print("The Regularized Cost is: ",costReg(theta1, theta2, input_size, hidden_size, num_labels, X, y_onehot, learning_rate))

## STEP_04: Achieve Backpropagation (to calculate gradient)
# 1. Use Backpropagation to calculate gradient;
# 2. Use gradient and the function in Library to calculate J(θ)min and get the θ;

# 01_Define Sigmoid Gradient (convenient for computing) [Sigmoid Gradient == the derivative(偏导数) of Sigmoid Function
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

# 02_Random Initialization
# np.random.random(size) 返回size大小的0-1随机浮点数
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24
# Mathmatically we should set -ε<θ<ε, here we make ε=0.12,this range makes sure that parameter is small enough
# It makes the algo more efficient

# 03_Achieve Backpropagation in a function:backprop()

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):     # Return J, theta1, theta2
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # run the Forward Propagation to get a1, z2, z3, and hypothesis function h;
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # Reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # Initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # Shape is (25, 401)
    delta2 = np.zeros(theta2.shape)  # Shape is (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # Perform Backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t       # The iteration process

    delta1 = delta1 / m
    delta2 = delta2 / m

    return J, delta1, delta2

# 04_Gradient Checking (ignore)
# if the Backpropagation is correct, the result in this part will be smaller than 10e-9

# 05_Regularize the whole Neural Networks
def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):      # Return J and gradient
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the Forward Propagation to get a1, z2, z3, and hypothesis function h;
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # Initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # Add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # Unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))     # 默认axis=0, 直接拼接到一起

    return J, grad

# 06_Use tools in Library to calculate J_min
from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backpropReg, x0=(params), args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
print("The fmin is: ",fmin)

# 07_Predict the accuracy & Visualize the hidden layer
X = np.matrix(X)
thetafinal1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
thetafinal2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

# Computer the predict accuracy by theta_min
a1, z2, a2, z3, h = forward_propagate(X, thetafinal1, thetafinal2 )
y_pred = np.array(np.argmax(h, axis=1) + 1)
print("y_pred is: ",y_pred)

from sklearn.metrics import classification_report   # Show the report
print(classification_report(y, y_pred))

# Visualize the hidden layer
hidden_layer = thetafinal1[:, 1:]
print("Shape of hidden layer is:",hidden_layer.shape)
fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(12, 12))
for r in range(5):
    for c in range(5):
        ax_array[r, c].matshow(np.array(hidden_layer[5 * r + c].reshape((20, 20))),cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()