import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Seaborn can be seen as a suppliment of matplotlib
# 在大多数情况下使用seaborn就能做出很具有吸引力的图, 而使用matplotlib就能制作具有更多特色的图

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

# Define Cost Function without Regularization
def cost(theta, X, y):
    """
    X: R(m*n), m records, n features
    y: R(m)
    theta : R(n), linear regression parameters
    """
    m = X.shape[0]      # Get the height of X
    inner = X @ theta - y       # The element for sum calculation, X @ theta equals to X.dot(theta)

    square_sum = inner.T @ inner    # square_sum为inner的每个元素平方之和，即二范数
    #  自身.T叉乘自身相当于实现了inner^2的同时，对inner^2实现了一次m的sum求和；
    cost = square_sum / (2 * m)     # Finish Cost Function computing
    return cost

# Define Regularized Cost Function
def costReg(theta, X, y, reg=1):        # reg means the learning rate λ
    m = X.shape[0]      # Get the height of X
    regularized_term = (reg / (2 * m)) * np.power(theta[1:], 2).sum()       # Finished the Regularization term

    return cost(theta, X, y) + regularized_term

# theta初始值为[1,1]，the test answer should be 303.993
theta = np.ones(X.shape[1])
print("The test answer of Regularized Cost Function = ",costReg(theta, X, y, 1))     # Test the costReg

# Define Gradient without Regularization
def gradient(theta, X, y):
    m = X.shape[0]      # Get the height of X
    inner = X.T @ (X @ theta - y)       # Scale: (m,n).T @ (m, 1) -> (n, 1)
    # Tricky: 在实现(h(x)-y)*xj的同时实现了sum求和运算

    return inner / m

# Define Regularized Gradient
def gradientReg(theta, X, y, reg):
    m = X.shape[0]      # Get the height of X

    regularized_term = theta.copy()      # Same shape as theta
    regularized_term[0] = 0     # Don't regularize intercept theta(theta0)
    regularized_term = (reg / m) * regularized_term

    return gradient(theta, X, y) + regularized_term     # Finish the calculation of Regualrized Gradient

print("The test result of Regularized Gradient (λ=1) = ",gradientReg(theta, X, y, 1))

## STEP_03: Find the final theta and Plot the h(x)
theta = np.ones(X.shape[1])
final_theta = opt.minimize(fun=costReg, x0=theta, args=(X, y, 0), method='TNC', jac=gradientReg, options={'disp': True}).x
print("The final theta is: ",final_theta)

b = final_theta[0]      # Intercept Y轴上的截距
m = final_theta[1]      # Slope 斜率

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
        X: feature matrix, (m, n+1)     # With incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.ones(X.shape[1])     # Initialize θ

    # train it
    res = opt.minimize(fun=costReg,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=gradientReg,
                       options={'disp': True})
    return res

training_cost, cv_cost = [], []

m = X.shape[0]      # Get the height of X
for i in range(1, m + 1):       # Let i be 1, 2, ..., m
    res = linear_regression(X[:i, :], y[:i], 0)     # Without Regularization
    # 对X任意列，第i行前(不包括第i行)
    tc = costReg(res.x, X[:i, :], y[:i], 0)     # Without Regularization
    cv = costReg(res.x, Xval, yval, 0)      # Without Regularization

    training_cost.append(tc)
    cv_cost.append(cv)      # Add at the rear

fig, ax = plt.subplots(figsize=(12,8))
plt.plot(np.arange(1, m+1), training_cost, label='training cost')
plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
plt.legend()
plt.show()      # Show the Underfitting Learning curve

## STEP_05: To tackle with Underfitting
# Add more features: Define Poly_Features
def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.values if as_ndarray else df

data = sio.loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex5/ex5/ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = map(np.ravel,[data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])

print("The poly_features with power=3 is: ",poly_features(X, power=3))      # Generate poly_features

# 1. Use the cost and gradient function above;
# 2. Expand features to 8 degree;
# 3. Apply nomalization to deal with x^n;
# 4. lambda=0

def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())

def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_feature(df).values

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)     # Add a column filled with 1

    return [prepare(x) for x in args]

X_poly, Xval_poly, Xtest_poly= prepare_poly_data(X, Xval, Xtest, power=8)

# Define a function to plot learning curve and h(x)
def plot_learning_curve(X, Xinit, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression(X[:i, :], y[:i], l=l)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].plot(np.arange(1, m + 1), training_cost, label='training cost')
    ax[0].plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    ax[0].legend()

    fitx = np.linspace(-50, 50, 100)
    fitxtmp = prepare_poly_data(fitx, power=8)
    fity = np.dot(prepare_poly_data(fitx, power=8)[0], linear_regression(X, y, l).x.T)

    ax[1].plot(fitx, fity, c='r', label='fitcurve')
    ax[1].scatter(Xinit, y, c='b', label='initial_Xy')

    ax[1].set_xlabel('water_level')
    ax[1].set_ylabel('flow')

# Use the function above to plot the learning curve and h(x)
plot_learning_curve(X_poly, X, y, Xval_poly, yval, l=0)
plt.show()      # The training cost is too low, Overfitting

## STEP_06: Choose the parameter lambda(λ) automatically
#   lambda will try: [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []

for l in l_candidate:
    res = linear_regression(X_poly, y, l)       # This lambda is l not 1

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)

    training_cost.append(tc)
    cv_cost.append(cv)
    theta = res.x
    print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))














