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

## STEP_05: Define the Batch Gradient Descent Algorithm
# NOTICE: The var of J(θ)isθ, not X or y. It means the guidline is we update the value of θ to update J(θ);
# How to inspect if the algorithm is well-operating: Print the value of J(θ) in each step.
# then see if it is decreasing and converge to a stable value.
# The final result will be used to predict "小吃店在35000及70000人城市规模的利润"

def gradientDescent(X, y, theta, alpha, iters):     # The defination of Batch Gradient Descent algorithm
    temp = np.matrix(np.zeros(theta.shape))     # Initialize a matrix filled with 0 (same scale with theta matrix)
    #NOTICE: 用法：zeros(shape, dtype=float, order='C')
    #返回来一个给定形状和类型的用0填充的数组；

    parameters = int(theta.ravel().shape[1])    # the number of parameter-θ
    cost = np.zeros(iters)  # Initialize an array containing the value of J(θ) in each iteration

    for i in range(iters):
        error = (X * theta.T) - y   # to simplify the expression below

        for j in range(parameters):
            term = np.multiply(error, X[:, j])  # iteration step of θj, same as part of the iteration fomular
            # NOTICE: method multiply():数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))    # same as the iteration fomular

        theta = temp    # same as the iteration fomular
        cost[i] = computeCost(X, y, theta)  # Calculate J(θ)

    return theta, cost
# this part achieves the update of Ѳ

## STEP_06: Initialize 2 additional vars - Learning Rate α & number of iterations
# let α be 0.01, iters be 1500
alpha = 0.01
iters = 1500

## STEP_07: Run the algorithm, predict and draw the graph
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)    # get the value of θ

predict1 = [1,3.5]*g.T
print("predict1:",predict1)
predict2 = [1,7]*g.T
print("predict2:",predict2)     # predict the profit of food stand with city scale in35000 & 70000 respectively

x = np.linspace(data.Population.min(), data.Population.max(), 100)  # Set horizontal ordinates(population)
f = g[0, 0] + (g[0, 1] * x)     # Set vertical ordinates(profit)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')     # H-O with x, V-O with f, 'r' means red colour, the name of this line is 'Prediction'
ax.scatter(data.Population, data.Profit, label='Traning Data')      # Show the scatter of traning data
ax.legend(loc=2)    # NOTICE: legend() have a parameter-loc, in order to control the location of legend(图例)
# example: plot.legend(loc=2), means第二象项(左上角)。loc can be set as 1,2,3,4
ax.set_xlabel('Population')
ax.set_ylabel('Profit')     # Name x&y label
ax.set_title('Predicted Profit vs. Population Size')    # Name title of the figure
plt.show()
# draw data graph







