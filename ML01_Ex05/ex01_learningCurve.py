import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Target: 1. Firstly we will make Regularized Linear Regression to a 水库的流出水量以及水库水位;
### 2. Secondly, we will discuss the question of Bias vs. Variance;

## STEP_01: Load Data and achieve Visualization
data = sio.loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex5/ex5/ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = map(np.ravel,[data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])
print("Shape of X is: ",X.shape, "Shape of y is: ",y.shape)
print("Shape of X_validation is: ",Xval.shape,"Shape of y_validation is: ",yval.shape)
print("Shape of Xtest is: ",Xtest.shape, " Shape of ytest is: ",ytest.shape)        # Check the shape of Data set

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X, y)
ax.set_xlabel('water_level')
ax.set_ylabel('flow')
plt.show()      # Finish Visualization

## STEP_02: Regularize the Cost Function & Gradient
# Insert x0 cols
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

# Define Cost Function
def cost(theta, X, y):
    """
    X: R(m*n), m records, n features
    y: R(m)
    theta : R(n), linear regression parameters
    """
    m = X.shape[0]
    inner = X @ theta - y  # R(m*1)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost

# Define Regularized Cost Function
def costReg(theta, X, y, reg=1):
    m = X.shape[0]
    regularized_term = (reg / (2 * m)) * np.power(theta[1:], 2).sum()

    return cost(theta, X, y) + regularized_term

# theta初始值为[1,1]，输出应该为303.993
theta = np.ones(X.shape[1])
print("The test answer of Regularized Cost Function = ",costReg(theta, X, y, 1))     # Test the costReg

# Define Gradient
def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)

    return inner / m

# Define Regularized Gradient
def gradientReg(theta, X, y, reg):
    m = X.shape[0]

    regularized_term = theta.copy()      # Same shape as theta
    regularized_term[0] = 0     # Don't regularize intercept theta(theta0)
    regularized_term = (reg / m) * regularized_term

    return gradient(theta, X, y) + regularized_term

print("The test result of gradientReg() = ",gradientReg(theta, X, y, 1))

## STEP_03: Find the final theta and Plot the h(x)
theta = np.ones(X.shape[1])
final_theta = opt.minimize(fun=costReg, x0=theta, args=(X, y, 0), method='TNC', jac=gradientReg, options={'disp': True}).x
print("The final theta is: ",final_theta)

b = final_theta[0]      # Intercept
m = final_theta[1]      # Slope

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(X[:,1], y, c='r', label="Training data")
plt.plot(X[:, 1], X[:, 1]*m + b, c='b', label="Prediction")
ax.set_xlabel('water_level')
ax.set_ylabel('flow')
ax.legend()
plt.show()

## STEP_04: Upgrade the Learning Algorithm
# 1.使用训练集的子集来拟合应模型; 2.在计算训练代价和验证集代价时，没有用正则化; 3.记住使用相同的训练集子集来计算训练代价;
def linear_regression(X, y, l=1):
    """linear regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=costReg,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=gradientReg,
                       options={'disp': True})
    return res

training_cost, cv_cost = [], []

m = X.shape[0]
for i in range(1, m + 1):
    res = linear_regression(X[:i, :], y[:i], 0)

    tc = costReg(res.x, X[:i, :], y[:i], 0)
    cv = costReg(res.x, Xval, yval, 0)

    training_cost.append(tc)
    cv_cost.append(cv)

fig, ax = plt.subplots(figsize=(12,8))
plt.plot(np.arange(1, m+1), training_cost, label='training cost')
plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
plt.legend()
plt.show()      # Show the Underfitting Learning curve










