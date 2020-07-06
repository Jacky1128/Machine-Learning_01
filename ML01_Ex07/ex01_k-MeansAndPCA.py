import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

### Target01: Apply K-means algorithm to a basic 2-D data set, then to use it to compress a image;
### Target02: Apply PCA algorithm to find the lower dimention expression of Facial image;

## T01_STEP01: Define a function to find the closest centroids and test it  (Minimize the c var in Distortion)
# Find 每条数据距离哪个类中心最近, 即c(i)
def find_closest_centroids(X, centroids):
    m = X.shape[0]      # Get the height of X (数据条数)
    k = centroids.shape[0]      # Get the height of controids (类的总数)
    idx = np.zeros(m)       # Initialize the index matrix

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)     # ** means 指数运算
            if dist < min_dist:
                min_dist = dist     # Let 'min_dist' be the smallest distence
                idx[i] = j      # Let the idx[i](which x(i) points to) be the type number j

    return idx      # Return the index matrix

# Test the function above
data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex7/ex7/ex7data2.mat')
X = data['X']

initial_centroids = initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])    # Artificially choose initial centroids

idx = find_closest_centroids(X, initial_centroids)
print("The index of closet centorids to x(i) are: ",idx[0:3])
# 输出与文本中的预期值匹配（记住我们的数组是从零开始索引的，而不是从一开始索引的，所以值比练习中的值低一个）
data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
print("Data2 is: ",data2.head())

sb.set(context="notebook", style="white")
sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
plt.show()

## T01_STEP02: Define a function to compute(move) the centroids  (Minimize the u var in Distortion)
# 计算类中心
def compute_centroids(X, idx, k):       # k is the number of clusters
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)   # Find the index of x(i) pointing to u(i)
        # np.where(condition),output index of 满足条件 (即非0) 的元素
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()

    return centroids        # Return the matrix of new uk

print("The updated cluster centroids(k=3) is: ",compute_centroids(X, idx, 3))

## T01_STEP03: Build & run the complete K-means algorithm then visualize the result
# Define Random Initialization function
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)    # 返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
    #如果没有写参数high的值，则返回[0,low)的值。

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids

# Define the complete K-means algorithm function
def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):      # The 2 loops
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids

initial_centroids = init_centroids(X, 3)    # Random initialize centroids(k=3)

idx, centroids = run_k_means(X, initial_centroids, 10)      # Let iteration times = 10
cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]

# Visualize
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 01')
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 02')
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 03')
ax.set_title("Data set01 after K-means algorithm")
ax.legend()
plt.show()

## T01_STEP04: Use K-means in image compression
## Find the most typical colours then let the raw 24 kinds of colours reflect to lower colour dimentions
from IPython.display import Image       # Load the Image waiting for compression
Image(filename='C:/Users/JackyWang28/Desktop/machine-learning-ex7/ex7/bird_small.png')







