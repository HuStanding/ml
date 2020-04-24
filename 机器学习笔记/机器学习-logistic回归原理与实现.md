

[toc]

# 一、什么是logistic回归？

logistic 回归又叫**对数几率回归**，适合数值型的二值型输出的拟合，它其实是一个分类模型，比如根据患者的医疗数据判断它是否能被治愈。

# 二、logistic回归数学原理与算法实现

我们考虑1个输入的$n$维数据$x=(x_1,x_2,\ldots,x_n)$，我们对输入数据进行线性加权得到
$$
g(x)=w_{0}+w_{1} x_{1}+\ldots+w_{n} x_{n}=w^{T}x \tag{1}
$$
前面说到，logistic回归用于而分类，假设得到的类别为0或者1，那么可以使用sigmoid函数处理输入，这个函数类似于阶跃函数但是又是连续型函数，看下这个函数长什么样

![sigmoid](https://tva1.sinaimg.cn/large/00831rSTly1gd8wzlk8aej318g0m8t9k.jpg)

$ \text{sigmod}(x)$ 其实衡量的是输入数据 $x$ 归属于类别 1 的概率，当 $x <0$ 的时候，$\text{sigmod}(x) < 0.5$ ，可以认为 $x$ 归属于类别 0 的概率较大，当  $x >0$ 的时候，$\text{sigmod}(x) > 0.5$，可以认为 $x$ 归属于类别 1 的概率较大。如果我们将线性加权得到的 $g(x)$ 作为 sigmoid 函数的输入，得到
$$
f(x)=\frac{1}{1+e^{-g(x)}}=\sigma(g(x))=\sigma(w^Tx)\tag{2}
$$
这样就得到了输入数据 $x$ 最终属于类别 1 的概率。

我们先考虑使用常规的均方差作为损失函数，这时候的损失函数为
$$
L(x)=\frac{1}{2}(y-f(x))^2=\frac{1}{2}\left(y-\sigma(w^Tx)\right)^2\tag{3}
$$
采用梯度下降的方法对 $w$ 进行更新，那么需要将损失函数对 $w$ 求导得到
$$
\frac{\partial L}{\partial w}=\left(y-\sigma(w^Tx)\right)\sigma'(w^Tx)x\tag{4}
$$
看到了吗？这里的梯度更新中包含了 $\sigma'(w^Tx)$ ，而通过 sigmod 函数可以发现，当 $\sigma(w^Tx)$ 位于 0 或者 1附近的时候，导数值基本趋近于 0，梯度收敛速度极慢。

所以在这种情况下我们可以考虑使用**交叉熵**作为损失函数。将$g(x)$作为输入数据$x$的输出，最上式做个简单的变换
$$
\ln\frac{f(x)}{1-f(x)}=w^Tx\tag{5}
$$
将$f(x)$视为类后验概率估计$P(y=1|x)$，则上式可以重写为
$$
\ln \frac{P(y=1 | x)}{P(y=0 | x)}=w^{T}x
$$
那么从而可以得到
$$
P(y=1 | x)=f(x)\\
P(y=0 | x)=1-f(x) \tag{6}
$$
上式可以合并为
$$
P(y | x,w)=[f(x)]^y \cdot [1-f(x)]^{1-y}\tag{7}
$$
设输入数据为$X=\left[\begin{array}& {x_{11}} & {x_{12}} & {\dots} & {x_{1 n}} \\ {x_{21}} & {x_{22}} & {\dots} & {x_{2 n}} \\ {\vdots} & {\vdots} & {\vdots} & {\dots} & {\vdots} \\  {x_{m 1}} & {x_{m 2}} & {\dots} & {x_{m n}}\end{array}\right]=\{x_1,x_2,\ldots,x_m\}$，$y=\left[\begin{array}{c}{y_{1}} \\ {y_{2}} \\ {\vdots} \\ {y_{m}}\end{array}\right]$，$x_i$表示第$i$个输入数据，上式的似然函数为
$$
L(w)=\prod_{i=1}^{m}\left[f\left(x_{i}\right)\right]^{y_{i}}\left[1-f\left(x_{i}\right)\right]^{1-y_{i}}\tag{8}
$$
然后我们的目标是求出使这一似然函数的值最大的参数估计，最大似然估计就是求出参数$w_0,w_1,\ldots,w_n$，使得上式取得最大值，对上式求导得到
$$
\ln L(w)=\sum_{i=1}^{m}\left(y_{i} \ln \left[f\left(x_{i}\right)\right]+\left(1-y_{i}\right) \ln \left[1-f\left(x_{i}\right)\right]\right)\tag{9}
$$

> 这里补充说明一下均方差和交叉熵损失的区别：均方差注重每个分类的结果，而交叉熵只注重分类正确的结果，所以交叉熵适合于分类问题，而不适合于回归问题，但是 logistic回归其实本质是 0-1 分类问题，所以这里依然适合作为 logistic 回归的损失函数。

## 2.1 梯度上升法估计参数

我们考虑$\ln L(w)$中间的一部分，对$w_k$求导得到
$$
\begin{equation}
\begin{split}
& {\left(y_{i} \ln \left[f\left(x_{i}\right)\right]+\left(1-y_{i}\right) \ln \left[1-f\left(x_{i}\right)\right]\right)^{\prime}} \\ 
& {=\frac{y_{i}}{f\left(x_{i}\right)} \cdot\left[f\left(x_{i}\right)\right]^{\prime}+\left(1-y_{i}\right) \cdot \frac{-\left[f\left(x_{i}\right)\right]^{\prime}}{1-f\left(x_{i}\right)}} \\ 
& {=\left[\frac{y_{i}}{f\left(x_{i}\right)}-\frac{1-y_{i}}{1-f\left(x_{i}\right)}\right] \cdot\left[f\left(x_{i}\right)\right]^{\prime}} \\ 
& {=\left(f\left(x_{i}\right)-y_{i}\right) g^{\prime}(x)} \\ 
& {=x_{i k}\left[f\left(x_{i}\right)-y_{i}\right]}
\end{split}
\end{equation}\tag{10}
$$
那么
$$
\frac{\partial \ln L\left(w_{k}\right)}{\partial w_{k}}=\sum_{i=1}^{m} x_{ik}\left[f\left(x_{i}\right)-y_{i}\right]=0\tag{11}
$$
我们使用**梯度上升法(Gradient ascent method)**求解参数$w$，其迭代公式为
$$
w=w+\alpha \nabla\ln L(w)\tag{11}
$$
梯度已经在上面计算过了，即
$$
\nabla\ln L(w)=\frac{\partial \ln L\left(w\right)}{\partial w}\tag{12}
$$
定义
$$
E=f(x)-y\tag{13}
$$
所以我们现在可以得到
$$
w=w+\alpha \sum_{i=1}^{m} x_{ik}\left[f\left(x_{i}\right)-y_{i}\right]=w+ \alpha X^TE\tag{14}
$$

代码实现

```python
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatrix, classLabels):
    dataMatrix = mat(dataMatrix)  # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    m, n = shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = ones((n, 1))
    weights_list = list()
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix*weights)  # matrix mult
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
        weights_list.append(weights)
    return weights, weights_list                                            

    
def plot_weights(weights_list):
    font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=12)
    fig = plt.figure(figsize=(8, 8))
    x = range(len(weights_list))
    w0 = [item[0, 0] for item in weights_list]
    w1 = [item[1, 0] for item in weights_list]
    w2 = [item[2, 0] for item in weights_list]
    plt.subplot(311)
    plt.plot(x, w0, 'r-', label="w0")
    plt.ylabel("w0")
    plt.subplot(312)
    plt.plot(x, w1, 'g-', label="w1")
    plt.ylabel("w1")
    plt.subplot(313)
    plt.plot(x, w2, 'b-', label="w2")
    plt.ylabel("w2")
    plt.xlabel("迭代次数", FontProperties=font)
    plt.show()


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y.transpose())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plot_sigmoid():
    x = arange(-60.0, 60.0, 1)
    y = sigmoid(x)
    fig = plt.figure(figsize=(8, 4))
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.plot(x, y.transpose())
    plt.show()


if __name__ == "__main__":
    data_mat, label_mat = loadDataSet()
    weights, weights_list = gradAscent(data_mat, label_mat)
    #plot_weights(weights_list)
    #plotBestFit(weights)
    #plot_sigmoid()
```

> 注意：在上程序的第45行使用的是$w_0+w_1x_1+w_2x_2=0$等式，而不是$w_0+w_1x_1+w_2x_2=1$，为什么呢？我们观察sigmoid函数可以发现0值是函数决策边界，而$g(x)=w_0+w_1x_1+w_2x_2$又是sigmoid函数的输入，所以另$g(x)=0$便可以得到分类边界线。

上述程序的运行结果如下图所示。

<img src="https://tva1.sinaimg.cn/large/006tNbRwly1g9jleh9thtj30xc0m8q3r.jpg" alt="logistic regression" style="zoom:50%;" />

迭代次数与最终参数的变化如下图所示。
![logistic iteration](https://tva1.sinaimg.cn/large/006tNbRwly1g9jo5ve15yj30m80m8ab6.jpg)

## 2.2 改进的随机梯度上升法

梯度上升算法在每次更新回归系数(最优参数)时，都需要遍历整个数据集。假设，我们使用的数据集一共有100个样本，回归参数有 3 个，那么dataMatrix就是一个100×3的矩阵。每次计算h的时候，都要计算dataMatrix×weights这个矩阵乘法运算，要进行100×3次乘法运算和100×2次加法运算。同理，更新回归系数(最优参数)weights时，也需要用到整个数据集，要进行矩阵乘法运算。总而言之，该方法处理100个左右的数据集时尚可，但如果有数十亿样本和成千上万的特征，那么该方法的计算复杂度就太高了。因此，需要对算法进行改进，我们每次更新回归系数(最优参数)的时候，能不能不用所有样本呢？一次只用一个样本点去更新回归系数(最优参数)？这样就可以有效减少计算量了，这种方法就叫做**随机梯度上升算法(Stochastic gradient ascent)**。

```python
def stocGradAscent(dataMatrix, classLabels, numIter=150):
    dataMatrix = mat(dataMatrix)
    labelMat = mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)                                                
    weights = np.ones((n,1))  
    weights_list = list()                                                    
    for j in range(numIter):                                           
        dataIndex = list(range(m))
        for i in range(m):           
            alpha = 4/(1.0+j+i)+0.01            
            randIndex = int(random.uniform(0,len(dataIndex)))              
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                   
            error = classLabels[randIndex] - h                               
            weights = weights + alpha * dataMatrix[randIndex].transpose() * error 
            del(dataIndex[randIndex])  
            weights_list.append(weights)                                      
    return weights, weights_list 
```

该算法第一个改进之处在于，alpha在每次迭代的时候都会调整，并且，虽然alpha会随着迭代次数不断减小，但永远不会减小到0，因为这里还存在一个常数项。必须这样做的原因是为了保证在多次迭代之后新数据仍然具有一定的影响。如果需要处理的问题是动态变化的，那么可以适当加大上述常数项，来确保新的值获得更大的回归系数。另一点值得注意的是，在降低alpha的函数中，alpha每次减少$1/(j+i)$，其中j是迭代次数，i是样本点的下标。第二个改进的地方在于跟新回归系数(最优参数)时，只使用一个样本点，并且选择的样本点是随机的，每次迭代不使用已经用过的样本点。这样的方法，就有效地减少了计算量，并保证了回归效果。

使用随机梯度上升法获得的分类结果如下图所示

<img src="https://tva1.sinaimg.cn/large/006tNbRwly1g9jvdhicofj30xc0m8aav.jpg" alt="Stochastic gradient ascent" style="zoom:50%;" />

迭代次数与最终参数的变化如下图所示，可以看到，在 8000 次以后，各参数基本趋于稳定，这个过程大约迭代了整个矩阵 80次，相比较于原先的 300 多次，大幅减小了迭代周期。

![Stochastic gradient iteration](https://tva1.sinaimg.cn/large/006tNbRwly1g9jvkkdadbj30m80m8ab2.jpg)

# 三、sklearn算法实现

`sklearn`实现 logistic 回归使用的是`linear_model`模型，还是使用上述的数据，代码如下

```python
def logistic_lib(dataMatrix, classLabels):
    from sklearn import linear_model
    model_logistic_regression = linear_model.LogisticRegression(solver='liblinear',max_iter=10)
    classifier = model_logistic_regression.fit(dataMatrix, classLabels)
    # 回归系数
    print(model_logistic_regression.coef_)
    # 截距
    print(model_logistic_regression.intercept_)
    # 准确率
    accurcy = classifier.score(dataMatrix, classLabels) * 100
    print(accurcy)
```

输出结果如下

```
[[ 2.45317293  0.51690909 -0.71377635]]
[2.45317293]
97.0
```




# 参考
[1] [CSDN博客-logistic回归原理解析--一步步理解](https://blog.csdn.net/lgb_love/article/details/80592147)
[2] [CSDN博客-logistic回归原理及公式推导](https://blog.csdn.net/AriesSurfer/article/details/41310525)
[3] 李航. 统计学习方法, 清华大学出版社

[4] [CSDN博客-logistic回归损失函数](https://blog.csdn.net/m0_37864814/article/details/94625639)