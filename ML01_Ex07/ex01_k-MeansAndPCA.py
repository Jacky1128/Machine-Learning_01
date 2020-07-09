import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

### Target01: Apply K-means algorithm to a basic 2-D data set, then to use it to compress a image;(降维)
### Target02: Apply PCA algorithm to find the lower dimention expression of Facial image;(降维)

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
##Image Compression understanding:
##1. Raw image is 128*128 3通道，将图像的宽、高压缩到一个维度，保留通道数为一个维度，最终为16384*3的数据量;
##2. 可以理解为这张图片一共有16384行数据，每行数据有3个特征;
##3. 然后对这些数据设置16个簇（对于原始图片，可以理解为分成了16块），通过kmeans算法得到16各簇中心点,将这16384行数据设置所属对应簇;
##4. 保存这张压缩图片，保存这16个簇中心数据，以及这16384行数数据对应类别即可，那么需要的数据量就是16384+16*3;

#01_load the image data
from IPython.display import Image       # Load the Image waiting for compression
Image(filename='C:/Users/JackyWang28/Desktop/machine-learning-ex7/ex7/bird_small.png')

# Load the Image data
image_data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex7/ex7/bird_small.mat')
print("The image data is: ",image_data)

A = image_data['A']
print("Shape of the matrix A in the image is: ",A.shape)        #  图像为128*128 3通道的图片
plt.imshow(A)
plt.show()

#02_Data preprocessing
# Normalize value ranges, because every data is between 0 and 255
A = A / 255.

# Reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))        # 重置矩阵大小，将行数和列数合并，通道为单独的一维
print("Shape of the reshaped X is: ",X.shape)

# Randomly initialize the centroids
initial_centroids = init_centroids(X, 16)       # Artificially let K=16, means there are 16 cluters

# Run the K-means algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)      # iterate 10 times

# Get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

# Map each pixel to the centroid value
X_recovered = centroids[idx.astype(int),:]      # X_recover is the matrix uk that all xi indicate to
print("Shape of X_recovered is: ",X_recovered.shape)
# dtype 用于查看数据类型, astype 用于转换数据类型

#03_Use kmeans module in scikit-learn package to compress the image
# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
print("Shape of the X after reshaped",X_recovered.shape)
plt.imshow(X_recovered)
plt.show()      # Compressed image with K-Means algorithm


## T02: 01. Apply PCA into a simple 2-D data set in order to learn how it works;
## 02. Apply PCA in a facial image. We can use smaller data to capture图像的“本质”;
## T02_01:
data = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex7/ex7/ex7data1.mat')
print("The 2-D data set is: ",data)

X = data['X']       # Get data X
# Visualize the data
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


def pca(X):
    # Mean Normalization of the features
    X = (X - X.mean()) / X.std()

    # Compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # Perform SVD in Library
    U, S, V = np.linalg.svd(cov)
    return U, S, V

U, S, V = pca(X)
print("The U matrix from PCA algorithm is: ",U)
# We will just get the first kth because we want the image dimention reduce to k-D
def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)     # Calculate U_reduced 点乘 X

Z = project_data(X, U, 1)       # Let k = 1
print("Z from PCA algorithm is: ",Z)

# We can also reconstructe the compressed data
def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)       # # Calculate Z 点乘 U_reduced.T

X_recovered = recover_data(Z, U, 1)
print("The recovered X is: ",X_recovered)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()      # The recovered data X

## T02_02: Apply the algorithm above in Facial image processing
# 给出的数据集包括5000张人脸图像，每张图像的大小为32*32灰度；
# 每一张图像的像素存储为1024维数值，数据集维度为 5000*1024；
# 选取前100张图像即可
faces = loadmat('C:/Users/JackyWang28/Desktop/machine-learning-ex7/ex7/ex7faces.mat')
X = faces['X']
print("Shape of faces_X is: ",X.shape)

def plot_n_image(X, n):     # Define Visualization function, n is number of images showed
    """ plot first n images
    n has to be a square number(完全平方数)
    """
    pic_size = int(np.sqrt(X.shape[1]))     # np.sqrt(B):求B的开方
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                    sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

plot_n_image(X, 100)
plt.show()      # Show the first 100 images

U, S, V = pca(X)
Z = project_data(X, U, 100)     # Let k = 100
X_recovered = recover_data(Z, U, 100)
# face = np.reshape(X_recovered[3,:], (32, 32))
# plt.imshow(face)
plot_n_image(X_recovered, 100)
plt.show()      # Show the first 100 images been compressed

# 我们读入的.mat文件画出来的是热量图，就是整体泛红，并不是期待的灰度图，
# 这是因为图像的存储方式可能与 cv库读取的方式是相同的，用plt函数画出来的图他们的通道不同
# 只需在其中一行代码中添加一个参数即可: 将 plot_n_image 函数的其中一行代码
# ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((32, 32))) 改为：
# ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((32, 32)),cmap='Greys_r')










