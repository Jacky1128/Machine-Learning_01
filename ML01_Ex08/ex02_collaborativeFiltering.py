import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

### Target: Achieve Collaborative Filtering Algorithm and apply it in a data set of movie ratings

## STEP_01: Load data set & visualize it
data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex8/ex8/ex8_movies.mat')
print("The data set is: ",data)

# Y is an array with levels from 1 -5 in scale of 电影数x用户数; R是包含指示用户是否给电影评分的二进制值的“指示符”数组
# They should have the same dimention
Y = data['Y']
R = data['R']
print("Shape of Y is: ",Y.shape, "; Shape of R is: ",R.shape)

# 通过平均排序Y来评估电影的平均评级。
print("The average rate is: ",Y[1,np.where(R[1,:]==1)[0]].mean())

# 通过将矩阵渲染成图像来尝试“可视化”数据以了解用户和电影的相对密度。
fig, ax = plt.subplots(figsize=(12,12))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
plt.show()

## STEP_02: Implement Collaborative Filtering algorithm将实施协同过滤的代价函数
# Define some operations functions
def serialize(X, theta):
    """序列化两个矩阵
    """
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    return np.concatenate((X.ravel(), theta.ravel()))

def deserialize(param, n_movie, n_user, n_features):
    """逆序列化"""
    return param[:n_movie * n_features].reshape(n_movie, n_features), param[n_movie * n_features:].reshape(n_user, n_features)

# recommendation fn
def cost(param, Y, R, n_features):
    """compute cost for every r(i, j)=1
    Args:
        param: serialized X, theta
        Y (movie, user), (1682, 943): (movie, user) rating
        R (movie, user), (1682, 943): (movie, user) has rating
    """
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    return np.power(inner, 2).sum() / 2

params_data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex8/ex8/ex8_movieParams.mat')
X = params_data['X']
theta = params_data['Theta']

users = 4
movies = 5
features = 3

X_sub = X[:movies, :features]
theta_sub = theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]

param_sub = serialize(X_sub, theta_sub)

cost(param_sub, Y_sub, R_sub, features)

param = serialize(X, theta)  # total real params

cost(serialize(X, theta), Y, R, 10)  # this is real total cost
## STEP_03: Achieve Gradient calculation
def gradient(param, Y, R, n_features):
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)  # (1682, 943)

    # X_grad (1682, 10)
    X_grad = inner @ theta

    # theta_grad (943, 10)
    theta_grad = inner.T @ X

    # roll them together and return
    return serialize(X_grad, theta_grad)

n_movie, n_user = Y.shape

X_grad, theta_grad = deserialize(gradient(param, Y, R, 10),n_movie, n_user, 10)
print("X_grad is: ",X_grad, "; theta_grad is: ",theta_grad)

## STEP_04: Regularize Cost function & Gradient
def regularized_cost(param, Y, R, n_features, l=1):
    reg_term = np.power(param, 2).sum() * (l / 2)

    return cost(param, Y, R, n_features) + reg_term


def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = l * param

    return grad + reg_term

print(regularized_cost(param_sub, Y_sub, R_sub, features, l=1.5))
print("Total regularized Cost is: ",regularized_cost(param, Y, R, 10, l=1))  # total regularized cost

n_movie, n_user = Y.shape

X_grad, theta_grad = deserialize(regularized_gradient(param, Y, R, 10),
                                                                n_movie, n_user, 10)

## STEP_05: 创建自己的电影评分，以便我们可以使用该模型来生成个性化的推荐。
movie_list = []
f = open('C:/Users/JackyWang28/Desktop/machine-learning-ex8/ex8/movie_ids.txt',encoding= 'ISO-8859-1')

for line in f:
    tokens = line.strip().split(' ')
    movie_list.append(' '.join(tokens[1:]))

movie_list = np.array(movie_list)

ratings = np.zeros((1682, 1))

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

# 将自己的评级向量添加到现有数据集中以包含在模型中。

Y = data['Y']
Y = np.append(ratings,Y, axis=1)  # now I become user 0
print("Shape of Y is: ",Y.shape)
R = data['R']
R = np.append( ratings != 0, R,axis=1)

movies = Y.shape[0]  # 1682
users = Y.shape[1]  # 944
features = 10
learning_rate = 10.

X = np.random.random(size=(movies, features))
theta = np.random.random(size=(users, features))
params = serialize(X, theta)

Y_norm = Y - Y.mean()
Y_norm.mean()

from scipy.optimize import minimize

fmin = minimize(fun=regularized_cost, x0=params, args=(Y_norm, R, features, learning_rate),
                method='TNC', jac=regularized_gradient)
print("The fmin is: ",fmin)

X_trained, theta_trained = deserialize(fmin.x, movies, users, features)

# 使用训练出的数据给出推荐电影

prediction = X_trained @ theta_trained.T
my_preds = prediction[:, 0] + Y.mean()
idx = np.argsort(my_preds)[::-1]  # Descending order
# top ten idx
my_preds[idx][:10]
for m in movie_list[idx][:10]:
    print("The recommend movies for you are: ",m)
