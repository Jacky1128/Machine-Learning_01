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
        data2['F' + str(i-j) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
#此处原答案错误较多，已经更正

data2.drop('Test 1', axis=1, inplace=True)      # Delete the "Text1" column and implace the origin data
data2.drop('Test 2', axis=1, inplace=True)      # Delete the "Text2" column and implace the origin data
# Make axis=1 if you want to delete columns

print(data2.head())

### STEP_03: Set Sigmoid Function g(z) & Cost Function J(θ) & Gradient


