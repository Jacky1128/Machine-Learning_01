import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#构建一个逻辑回归模型来预测某个学生是否被大学录取。
#设想你是大学相关部分的管理者，想通过申请学生两次测试的评分来决定他们是否被录取。
#现拥有之前申请学生的可以用于训练逻辑回归的训练样本集。对于每一个训练样本，你有他们两次测试的评分和最后是被录取的结果。

### STEP_01: Get & show the data set we've got
path = r'C:\Users\JackyWang28\Desktop\ex2data1.txt'     # Practical point "add 'r'" before the path
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(data.head())      # Show the origin data set

# Draw a scatter below, H-O is the score of exam1, V-O is the score of exam2;
# Admitted >0, don't admitted <0, use different colour to differentiate them
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
#NOTICE: pandas库中的isin()函数用于数据筛选; 接受一个列表，判断该列中元素是否在列表中，多用于要选择某列等于多个数值或者字符串时。
#data[data[‘admitted’].isin([‘1’])]选取admitted列值为1的所有行，等价于data[data[‘admitted’] == 1]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()     # Acquiescently put legend to the best area
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()      # Finish plotting the scatter graph

### STEP_02: Set Sigmoid Function g(z) & Cost Function J(θ) & Gradient
# Define Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define Cost Function
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))       #First step
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))     #Second step
    return np.sum(first - second) / (len(X))        #Finished

# Initialize X, y, θ

data.insert(0, 'Ones', 1)       # Almost all the same as Ex01
cols = data.shape[1]        # Get width
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.zeros(3)     # Shape as 1 row 3 cols

X = np.array(X.values)
y = np.array(y.values)      # Change X，y's type
print(X.shape, theta.shape, y.shape)       # Check the width

# Use origin θ to calculate J(θ)
print(cost(theta, X, y))


# Define a function to Calculate Gradient(NOT Gradient descent) without update θ
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    # ravel（）返回的是视图，意味着改变元素的值会影响原始数组；
    # flatten（）返回的是拷贝，意味着改变元素的值不会影响原始数组。

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad

### STEP_03: Use library to calculate the final θ automatically; We don't have to set iters and learning rate ourselves;
### the function from library will bring the best answer directly

import scipy.optimize as opt        # Use this library to calculate theta_min
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print(result)
#NOTICE: func：优化的目标函数（此例中为代价函数); fprime：提供优化函数func的梯度函数，不然优化函数func必须返回函数值和梯度;
# x0(array_like)：初值（此例中为θ); approx_grad :如果设置为True，会给出近似梯度; args：元组，是传递给优化函数的参数

# 返回值: x(ndarray): The solution.
# nfeval(int): The number of function evaluations.
# rc(int): Return code, see below

### STEP_04: Use the answer θ to calculate J(θ) and draw the graph
# Calculate J(θ)
print(cost(result[0], X, y))

# Draw the graph
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = ( - result[0][0] - result[0][1] * plotting_x1) / result[0][2]     # result[0][0] indicates θ0
# result[0][1] indicates θ1 ... plotting_h1 here indicate x2 in "θ0 + θ1x1 + θ2x2 = 0"

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Prediction')
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()      # Acquiescently put legend to the best area

ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()      # Logistic Regression without Regularization end

### STEP_05: Appraise the model above
# method_01: Use a new case to calculate the hypothesis function
def hfunc1(theta, X):   # Define the hypothesis function h(x)
    return sigmoid(np.dot(theta.T, X))      # np.dot是执行矩阵机关枪乘法, 而*乘法则是对应位置相乘，结果形状一样


# method_02: Define a predict function to calculate the accuracy
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

# Predict accuracy
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))







