import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## STEP_01: Get & show the data set 2
path = r'C:\Users\JackyWang28\Desktop\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())     # Show the data set we've got

## STEP_02: Dealing with Data by Mean Normalization
# size-var is 1000 times of bedrooms-var,hence we can use Mean Normalization to make Gradient Descenmt run faster
# particular operation : (每类特征-该特征平均值)/其标准差
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())     # Show the data set after operation


## STEP_03: [Choice1] Apply Gradient Descent algorithm
# Define 2 functions below
def computeCost(X, y, theta):  # The defination of the cost function J(Ѳ)，X is a matrix
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):     # The defination of Batch Gradient Descent algorithm
    temp = np.matrix(np.zeros(theta.shape))     # Initialize a matrix filled with 0 (same scale with theta matrix)
    #NOTICE: zeros(shape, dtype=float, order='C')
    #Return a all-zero array given particular shape and type

    parameters = int(theta.ravel().shape[1])    # the number of parameter-θ
    cost = np.zeros(iters)  # Initialize an array containing the value of J(θ) in each iteration

    for i in range(iters):
        error = (X * theta.T) - y   # to simplify the expression below

        for j in range(parameters):
            term = np.multiply(error, X[:, j])  # iteration step of θj, same as part of the iteration fomular
            # NOTICE: method multiply():array and matrix 对应位置相乘
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))    # same as the iteration fomular

        theta = temp    # same as the iteration fomular
        cost[i] = computeCost(X, y, theta)  # Calculate J(θ)

    return theta, cost
# this part achieves the update of Ѳ

data2.insert(0, 'Ones', 1)      # means column No.0, named as 'Ones', all valued as 1; actually the x0 in theory

cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))       # All the same as exercise2.2

g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)         # Run the algorithm
print(g2)

## cannot run in that the Learning Rate α is not defined

## STEP_03: [Choice2] Apply Normal equation method
def normalEqn(X, y):        # The defination of Normal Equation method
    theta = np.linalg.inv(X.T@X)@X.T@y      #X.T@X is the same as X.T.dot(X)
    return theta




