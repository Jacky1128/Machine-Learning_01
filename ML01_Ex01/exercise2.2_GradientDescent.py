import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# output the data set
path = r'C:\Users\JackyWang28\Desktop\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())

# draw the data graph
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()


def computeCost(X, y, theta):  # The defination of the cost function J(Ѳ)，X is a matrix
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

############################################################### achieve Gradient Descent below

## STEP_01: Add a new column about x(x0)=1 named 'Ones' to update theta0
data.insert(0, 'Ones', 1)   # means column No.0, named as 'Ones', all valued as 1; actually the x0 in theory

## STEP_02: Initialize the matrix X and y
cols = data.shape[1]    # use method shape[] to get the shape of the matrix;
# NOTICE:在矩阵中，[0]就表示行数，[1]则表示列数。因此data.shape[1]表实读取data这个矩阵的宽度；

X = data.iloc[:, :-1]   # means: 任意行，从列索引0到倒数第二列，即X剔除了data set中y的那一列
y = data.iloc[:, cols - 1:cols]     # means: 任意行，只取最后一列，即y只取data set中y的那一列
# NOTICE:iloc 是基于“位置”的Dataframe操作，基于整数的下标来进行数据定位/选择；语法data.iloc[<row selection>, <column selection>]


## STEP_03: Inspect whether the matrix X (training set) and y (目标变量) are correct or not
print(X.head())    # the method head() only get first 5 rows
print(y.head())

## STEP04: cost function can just operate numpy matrix, so here we change the form of X&y, then initialize theta
# NOTICE:numpy.matrix(data,dtype,copy):返回一个矩阵，dtype:为data的type；copy:为bool类型
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

# see the dimention
print(X.shape)
print(y.shape)
print(theta.shape)

# Calculate the cost function J(θ)
print(computeCost(X,y,theta))

## STEP_05: Achieve Gradient Descent
# NOTICE: The var of J(θ)isθ, not X or y. It means the guidline is we update the value of θ to update J(θ);
# How to inspect if the algorithm is well-operating: Print the value of J(θ) in each step.
# then see if it is decreasing and converge to a stable value.
# The final result will be used to predict "小吃店在35000及70000人城市规模的利润"

def gradientDescent(X, y, theta, alpha, iters):     # The defination of the gradient descent algorithm
    temp = np.matrix(np.zeros(theta.shape))     #NOTICE: 用法：zeros(shape, dtype=float, order='C')
    #返回来一个给定形状和类型的用0填充的数组；

    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost
# this part achieves the update of Ѳ







