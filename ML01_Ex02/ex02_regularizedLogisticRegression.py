import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 实现加入正则项提升逻辑回归算法。 设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果
# 测试结果决定是否芯片要被接受或抛弃。你有一些历史数据，帮助你构建一个逻辑回归模型。

### STEP_01: Get & show the data set we've got
path =  r'C:\Users\JackyWang28\Desktop\ex2data2.txt'
data_init = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
print(data_init.head())

# Draw a scatter below, H-O is the score of exam1, V-O is the score of exam2;
# Accepted >0, don't accepted <0, use different colour to differentiate them
positive2 = data_init[data_init['Accepted'].isin([1])]
negative2 = data_init[data_init['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['Test 1'], positive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()     # Acquiescently put legend to the best area
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()      # Finish plotting the scatter graph

# This data set cannot be separate with a straight line
# so we implement the Regularizaed Logistic Regression

### STEP_02: Set more features
degree = 6      # provide up to 6 degrees for each pair of x1&x2
data2 = data_init
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree+1):
    for j in range(0, i+1):
        data2['F' + str(i-j) + str(j)] = np.power(x1, i-j) * np.power(x2, j)    # set new cols from F11 to F66
# Fmn indicates x1^m*x2^n

data2.drop('Test 1', axis=1, inplace=True)      # Delete the "Text1" column and replace the origin data
data2.drop('Test 2', axis=1, inplace=True)      # Delete the "Text2" column and replace the origin data
# Make axis=1 if you want to delete columns

print(data2.head())

### STEP_03: Set Sigmoid Function g(z) & Cost Function J(θ) & Gradient
def sigmoid(z):     # Define Sigmoid Function
    return 1 / (1 + np.exp(-z))

def costReg(theta, X, y, learningRate):     # Define the Regularized Cost Function J(θ)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradientReg(theta, X, y, learningRate):     # Define Gradient after regularization
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)     # Initialize grad
    # ravel（）返回的是视图，意味着改变元素的值会影响原始数组；
    # flatten（）返回的是拷贝，意味着改变元素的值不会影响原始数组。

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)     # Calculate θ0
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])     # Calculate θi, i>0

    return grad

### STEP_04: Use library to calculate θmin, use predict function to predict the accuracy of and plot the graph
# Initialize X，y，θ
cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]
theta2 = np.zeros(cols-1)

# Change data type
X2 = np.array(X2.values)
y2 = np.array(y2.values)

# Set the Learning Rate λ
learningRate = 1        # Let it be 1

# Calculate the beginning cost function
print("The beginning Cost of the training set is: ",costReg(theta2, X2, y2, learningRate))

# Use Library to calculate θ
import scipy.optimize as opt        # Use this library to calculate theta_min
result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
# Regularized Cost function owns 3 parameters shown in 'args'
print("theta_min = ",result2[0])

# Define predict function
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

# Predict the accuracy
theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('The accuracy = {0}%'.format(accuracy))

# Plot the graph
def hfunc2(theta, x1, x2):      # Define the hypothesis function
    temp = theta[0][0]      # means θ0
    place = 0       # a counter
    for i in range(1, degree+1):
        for j in range(0, i+1):
            temp+= np.power(x1, i-j) * np.power(x2, j) * theta[0][place+1]      # Show theta.T * X
            place+=1
    return temp

def find_decision_boundary(theta):      # Define a Function to find Decision boundary by inputing theta_min
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)       # zip()压缩， zip(*)解压
    h_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    h_val['hval'] = hfunc2(theta, h_val['x1'], h_val['x2'])

    decision = h_val[np.abs(h_val['hval']) < 2 * 10**-3]
    return decision.x1, decision.x2

#Draw the graph
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['Test 1'], positive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x, y = find_decision_boundary(result2)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()      #End

### STEP_05: Change the Learning Rate to see the result
# Case 1: Learning Rate λ = 0, Overfitting
learningRate2 = 0
result3 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate2))

# Plot the graph with λ = 0
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['Test 1'], positive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x, y = find_decision_boundary(result3)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()

# Case 2: Learning Rate λ = 100, Underfitting
learningRate3 = 100
result4 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate3))

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['Test 1'], positive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x, y = find_decision_boundary(result4)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()





