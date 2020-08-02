[toc]

# 一、为什么要进行数据降维

机器学习领域中所谓的降维就是指采用某种映射方法，将原高维空间中的数据映射到低维度的空间中。之所以要进行数据降维，是因为在原始的高维数据中，存在很多冗余以及噪声信息，通过数据降维，我们可以减少冗余信息，提高识别的精度，同时降低维度也可以提升机器学习的速度。

# 二、原理

PCA 全称为**主成分分析方法**(Principal Component Analysis)，它的目标是通过某种线性投影，将高维的数据映射到低维的空间中表示，并期望在所投影的维度上数据的方差最大，以此使用较少的数据维度，同时保留住较多的原数据点的特性。

举个🌰，下图中的数据为 2 维，现在想只通过 1 个维度来表示这堆数据，通过将数据投影到 z 轴上，原始的点$\{x_1,x_2\}$在新的 z 轴上的数据为$z_1$。

<img src="https://tva1.sinaimg.cn/large/0082zybply1gbq5ujphfyj312i0k479u.jpg" alt="image-20200209150316294" style="zoom:50%;" />

那么怎么找到这个投影轴呢？这里的过程比较复杂，感兴趣的同学可以看 [[PCA的数学原理(转)]](https://zhuanlan.zhihu.com/p/21580949)，主要的过程就是通过协方差矩阵来求解特征向量从而获取降维后的轴。

假设原始数据表示为 $X \in \R^{m×n}$，数据维度为 $n$ ，PCA 算法的流程如下：

1. **均值标准化**

获取每个维度的均值，设 $\mu_j$ 为第 $j$ 个维度的均值，则
$$
\mu_j=\frac{1}{m}\sum_{i=1}^{m}x_{j}^{(i)}\tag{1}
$$
再对原始的数据进行替换，
$$
x_{j}=x_{j}-\mu_j \tag{2}
$$

2. **求解协方差矩阵**

经过均值标准化之后的数据的协方差矩阵为
$$
\Sigma=X^TX\tag{3}
$$

3. **获取特征向量**

一般来说， $\Sigma$ 会有 $n$ 个特征值，对应 $n$ 个特征向量，如果需要将原始数据从 $n$ 维降低到 $k$ 维，则只需要选取特征值最大的 $k$ 个特征值对应的特征向量即可，我们将其设为 $U$。

4. **降低数据维度**

低维数据可以表示为
$$
Z=XU \in \R^{m×k}\tag{4} 
$$

 这样就将原始的 $n$ 维数据降低为 $k$ 维。，这是如果想恢复原始数据怎么办？可以恢复部分维度，被压缩的部分是找不回来的，通过压缩后的数据还原到原始数据的公式为
$$
X_{\text{approx}}=ZU^T+\mu\tag{5}
$$
举个🌰，假设原始数据为$X=\left(\begin{array}{cc}{-1} & {-2} \\ {-1} & {0} \\ {0} & {0} \\ {2} & {1} \\ {0} & {1}\end{array}\right)$，维度为 $n=2$，下面我们根据上述过程计算 1 个维度下的值。

首先计算均值，我们发现每列数据的均值为 0，那么则可以直接进行协方差矩阵的计算
$$
\Sigma=\frac{1}{5}\left(\begin{array}{ccccc}{-1} & {-1} & {0} & {2} & {0} \\ {-2} & {0} & {0} & {1} & {1}\end{array}\right) \left(\begin{array}{cc}{-1} & {-2} \\ {-1} & {0} \\ {0} & {0} \\ {2} & {1} \\ {0} & {1}\end{array}\right)=\left(\begin{array}{cc}{\frac{6}{5}} & {\frac{4}{5}} \\ {\frac{4}{5}} & {\frac{6}{5}}\end{array}\right)
$$
然后求其特征值和特征向量，具体求解方法不再详述，可以参考相关资料。求解后特征值为
$$
\lambda_{1}=2, \lambda_{2}=2 / 5
$$
其对应的特征向量分别是
$$
c_{1}=\left(\begin{array}{l}{1} \\ {1}\end{array}\right), c_{2}=\left(\begin{array}{c}{-1} \\ {1}\end{array}\right)
$$
其中对应的特征向量分别是一个通解，$c_{1}$ 和 $c_{2}$ 可取任意实数。那么标准化后的特征向量为
$$
\left(\begin{array}{c}{1 / \sqrt{2}} \\ {1 / \sqrt{2}}\end{array}\right),\left(\begin{array}{c}{-1 / \sqrt{2}} \\ {1 / \sqrt{2}}\end{array}\right)
$$
由于需要降低到 1 维，所以我们取特征值 $\lambda_1$ 对应的特征向量作为矩阵 $U=\left(\begin{array}{c}{1 / \sqrt{2}} \\ {1 / \sqrt{2}}\end{array}\right)$，降维后的数据为
$$
Z =\left(\begin{array}{cc}{-1} & {-2} \\ {-1} & {0} \\ {0} & {0} \\ {2} & {1} \\ {0} & {1}\end{array}\right)\left(\begin{array}{c}{1 / \sqrt{2}} \\ {1 / \sqrt{2}}\end{array}\right)=\left(\begin{array}{c}-3/\sqrt{2} \\ -1/\sqrt{2} \\ 0 \\ 3/\sqrt{2} \\ 1/\sqrt{2}\end{array}\right)
$$
注意⚠️：通过前面的方法我们知道还需要手动设置一个 $k$ 值，那么怎么选择最优的 $k$ 值呢，一般来说，选取的 $k$ 值通常要保留 99% 的方差，$k$ 值的选取可以参考下面的过程：

> 1. $k=1 \to n-1$ 
>
> 2. 通过式 $(3)、(4)、(5)$计算 $U,z^{(1)},z^{(2)},\ldots,z^{(m)},x_{\text{approx}}^{(1)},x_{\text{approx}}^{(2)},\ldots,x_{\text{approx}}^{(m)}$
>
> 3. 校对是否满足以下条件
>     $$
>     \frac{\frac{1}{m} \sum_{i=1}^{m}\left\|x^{(i)}-x_{\text {approx}}^{(i)}\right\|^{2}}{\frac{1}{m} \sum_{i=1}^{m}\left\|x^{(i)}\right\|^{2}}\leq0.01
>     $$
>  如果满足上述条件，则可以选择该 $k$ 

#  三、实现

## 3.1 Python 手动实现

```python
'''
@Author: huzhu
@Date: 2019-11-20 09:18:15
@Description: 
'''
import numpy as np
import matplotlib.pyplot as plt 

def load_data(file_name, delim='\t'):
    fr = open(file_name)
    str_arr = [line.strip().split(delim) for line in fr.readlines()]
    dat_arr = [list(map(float,line)) for line in str_arr]
    return np.mat(dat_arr)

def pca(data_mat, topNfeat = 999999):
    '''
    @description: PCA
    @return: low_data_mat, recon_mat
    '''
    mean_val = np.mean(data_mat, axis = 0)
    mean_removed = mean_val - data_mat
    # get the covrariance matrix
    cov_mat = np.cov(mean_removed, rowvar=0)
    # get the eigenvalue and eigenvector
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # sort, sort goes smallest to largest
    eigen_val_ind = np.argsort(eigen_vals)
    # cut off unwanted dimensions
    eigen_val_ind = eigen_val_ind[:-(topNfeat+1):-1]
    print(eigen_val_ind)
    # reorganize eig vects largest to smallest
    red_eigen_vecs = eigen_vecs[:,eigen_val_ind] 
    print(red_eigen_vecs)
    # low dimension data
    low_data_mat = mean_removed * red_eigen_vecs
    # transfor low data to original dimension
    recon_mat = (low_data_mat * red_eigen_vecs.T) + mean_val
    return low_data_mat, recon_mat

if __name__ == '__main__':
    data_mat = load_data("testSet.txt")
    low_data_mat, recon_mat = pca(data_mat, 1)
    plt.figure()
    plt.scatter(data_mat[:,0].flatten().A[0], data_mat[:,1].flatten().A[0], marker='^', s = 90)
    plt.scatter(recon_mat[:,0].flatten().A[0], recon_mat[:,1].flatten().A[0], marker='o', s = 50, c = "red")
    plt.show()
```

降维后的结果如下图所示。

<img src="https://tva1.sinaimg.cn/large/0082zybply1gbq7igvksyj30zk0qoq4g.jpg" alt="PCA" style="zoom:50%;" />

## 3.2 库函数实现

```python
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -2], [-1, 0], [0, 0], [2, 1], [0, 1]])
pca = PCA(n_components=1)
newX = pca.fit_transform(X)
print(newX)
print(pca.explained_variance_ratio_)
```

结果为

```
[[ 2.12132034]
 [ 0.70710678]
 [-0.        ]
 [-2.12132034]
 [-0.70710678]]
[0.83333333]
```

可以看到和我们上述举例的数据相同（符号相反是求解特征向量的时候符号不同所导致的）。

完整代码和数据可以参考 [[我的 github]](https://github.com/HuStanding/nlp-exercise/tree/master/pca)。

# 四、参考

[1] https://zhuanlan.zhihu.com/p/21580949

[2] https://www.jianshu.com/p/8642d5ea5389