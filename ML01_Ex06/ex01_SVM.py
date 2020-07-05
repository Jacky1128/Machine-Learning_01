import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

### Target: Use SVM with Gaussian kernel to build Spam Classifier

## STEP_01: Load and visualize raw data
raw_data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex6/ex6/ex6data1.mat')
data = pd.DataFrame(raw_data.get('X'), columns=['X1', 'X2'])
data['y'] = raw_data.get('y')

print(data.head())

def plot_init_data(data, fig, ax):
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]

    ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
    ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')

fig, ax = plt.subplots(figsize=(12, 8))
plot_init_data(data, fig, ax)
ax.set_title('Data Set 1')
ax.legend()
plt.show()      # Visualized

## STEP_02: Apply Linear kernel SVM Classification algorithm
# Let C = 1
from sklearn import svm
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=10000)
svc.fit(data[['X1', 'X2']], data['y'])

print("The performance of Linear kernel SVM is: ",svc.score(data[['X1', 'X2']], data['y']))
# model.predict(x_test)  # 输出类别
# model.predict_proba(x_test)  # 输出分类概率
# model.predict_log_proba(x_test)  # 输出分类概率的对数
# In sklearn, 我们可以使用完全一样的接口来实现不同的机器学习算法
#   1. 数据加载和预处理
#   2. 定义分类器（回归器等等），譬如svc = svm.svc()
#   3. 用训练集对模型进行训练，只需调用fit方法，svc.fit(X_train, y_train)
#   4. 用训练好的模型进行预测：y_pred=svc.predict(X_test)
#   5. 对模型进行性能评估：svc.score(X_test, y_test)
#   模型评估中，可以通过传入一个score参数来自定义评估标准，该函数的返回值越大代表模型越好

## STEP_03: Visualize Dicision Boundary
def find_decision_boundary(svc, x1min, x1max, x2min, x2max, diff):
    x1 = np.linspace(x1min, x1max, 1000)
# 在默认情况下，linspace函数可以生成元素为50的等间隔数列。而前两个参数分别是数列的开头与结尾。
# 如果写入第三个参数，可以制定数列的元素个数。
    x2 = np.linspace(x2min, x2max, 1000)

    cordinates = [(x, y) for x in x1 for y in x2]
    x_cord, y_cord = zip(*cordinates)
    c_val = pd.DataFrame({'x1': x_cord, 'x2': y_cord})
    c_val['cval'] = svc.decision_function(c_val[['x1', 'x2']])

    decision = c_val[np.abs(c_val['cval']) < diff]

    return decision.x1, decision.x2

x1, x2 = find_decision_boundary(svc, 0, 4, 1.5, 5, 2 * 10**-3)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x1, x2, s=10, c='r',label='Boundary')
plot_init_data(data, fig, ax)
ax.set_title('Linear kernel SVM (C=1) Decision Boundary in Data Set 1')
ax.legend()
plt.show()


## STEP_04: Change Linear kernel SVM into Gaussian kernel SVM (Given parameters manually) in another data set
raw_data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex6/ex6/ex6data2.mat')

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

fig, ax = plt.subplots(figsize=(12,8))
plot_init_data(data, fig, ax)
ax.set_title('Data Set 2')
ax.legend()
plt.show()

svc = svm.SVC(C=100, gamma=10, probability=True)

svc.fit(data[['X1', 'X2']], data['y'])
print("The performance of Gaussian kernel SVM is: ",svc.score(data[['X1', 'X2']], data['y']))

x1, x2 = find_decision_boundary(svc, 0, 1, 0.4, 1, 0.01)
fig, ax = plt.subplots(figsize=(12,8))
plot_init_data(data, fig, ax)
ax.scatter(x1, x2, s=10, c='#00CED1', label='Boundary')
ax.set_title('Gaussian kernel SVM (C=100) Decision Boundary in Data Set 2')
plt.show()

## STEP_05: Use data set 3(training & cv set) to find parameters(C and σ) for SVM automatically
## Candidate values are [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
raw_data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex6/ex6/ex6data3.mat')

X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y'].ravel()
yval = raw_data['yval'].ravel()

fig, ax = plt.subplots(figsize=(12,8))
data = pd.DataFrame(raw_data.get('X'), columns=['X1', 'X2'])
data['y'] = raw_data.get('y')
plot_init_data(data, fig, ax)
ax.set_title('Data Set 3')
plt.show()

# Select best parameters automatically
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0      # Initialize
best_params = {'C': None, 'gamma': None}

for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X, y)
        score = svc.score(Xval, yval)

        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma

print("The best performance with selected parameters is: ",best_score, "; The best parameters are: ",best_params)
# Output the best score value, C and gamma

svc = svm.SVC(C=best_params['C'], gamma=best_params['gamma'])
svc.fit(X, y)

x1, x2 = find_decision_boundary(svc, -0.6, 0.3, -0.7, 0.6, 0.005)
fig, ax = plt.subplots(figsize=(12,8))
plot_init_data(data, fig, ax)
ax.scatter(x1, x2, s=10)
ax.set_title('Complete version Gaussian kernel SVM Decision Boundary in Data Set 3')
plt.show()


## STEP_06: Use SVM algorithm to build Spam Classifier
##  Deal with Emails (to get data with SVM friendly form)extract features;
##  这个任务是将字词映射到为练习提供的字典中的ID）
## 而其余的预处理步骤（如HTML删除，词干，标准化等）已经完成。 我们就直接读取预先处理好的数据就可以了。

spam_train = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex6/ex6/spamTrain.mat')
spam_test = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex6/ex6/spamTest.mat')
print("The spam training set is: ",spam_train)

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()
print("Shape of training X is:",X.shape, ";Shape of training X is:",y.shape)
print("Shape of test X is:",Xtest.shape, "Shape of test y is:",ytest.shape)
# 每个文档已经转换为一个向量，其中1,899个维对应于词汇表中的1,899个单词。 它们的值为二进制，表示文档中是否存在单词。

svc = svm.SVC()
svc.fit(X, y)       # Training the SVC algorithm

print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))

## Visualize the result
kw = np.eye(1899)
kw[:3,:]
spam_val = pd.DataFrame({'idx':range(1899)})
spam_val['isspam'] = svc.decision_function(kw)
spam_val['isspam'].describe()

decision = spam_val[spam_val['isspam'] > -0.55]
path =  'C:/Users/JackyWang28/Desktop/machine-learning-ex6/ex6/vocab.txt'
voc = pd.read_csv(path, header=None, names=['idx', 'voc'], sep = '\t')
voc.head()
spamvoc = voc.loc[list(decision['idx'])]
print(spamvoc)

