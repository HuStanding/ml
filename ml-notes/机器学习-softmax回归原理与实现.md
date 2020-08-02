[toc]

# 一、什么是 softmax 回归？

**softmax 回归**(softmax regression)其实是 logistic 回归的一般形式，logistic 回归用于二分类，而 softmax 回归用于多分类，关于  logistic 回归可以看我的这篇博客👉[[机器学习-logistic回归原理与实现]](https://zhuanlan.zhihu.com/p/95132284)。

对于输入数据$\{(x_1,y_1),(x_2,y_2),\ldots,(x_m,y_m)\}$有 $k$ 个类别，即$y_i \in \{1,2,\ldots,k\}$，那么 softmax 回归主要估算输入数据 $x_i$ 归属于每一类的概率，即
$$
h_{\theta}\left(x_i\right)=\left[\begin{array}{c}{p\left(y_i=1 | x_i ; \theta\right)} \\ {p\left(y_i=2 | x_i ; \theta\right)} \\ {\vdots} \\ {p\left(y_i=k | x_i ; \theta\right)}\end{array}\right]=\frac{1}{\sum_{j=1}^{k} e^{\theta_{j}^{T} x_i}}\left[\begin{array}{c}{e^{\theta_{1}^{T} x_i}} \\ {e^{\theta_{2}^{T} x_i}} \\ {\vdots} \\ {e^{\theta_{k}^{T} x_i}}\end{array}\right]\tag{1}
$$
其中，$\theta_1,\theta_2,\ldots,\theta_k \in \theta$是模型的参数，乘以$\frac{1}{\sum_{j=1}^{k} e^{\theta_{j}^{T} x_i}}$是为了让概率位于[0,1]并且概率之和为 1，softmax 回归将输入数据 $x_i$ 归属于类别 $j$ 的概率为
$$
p\left(y_i=j | x_i ; \theta\right)=\frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\tag{2}
$$
上面的式子可以用下图形象化的解析(来自台大李宏毅《一天搞懂深度学习》)。

![img](https://tva1.sinaimg.cn/large/006tNbRwly1g9yc7t1ye4j30jg0awq4j.jpg)

# 二、原理

## 2.1 梯度下降法参数求解

softmax 回归的参数矩阵 $\theta$ 可以记为
$$
\theta=\left[\begin{array}{c}{\theta_{1}^{T}} \\ {\theta_{2}^{T}} \\ {\vdots} \\ {\theta_{k}^{T}}\end{array}\right]\tag{3}
$$
定义 softmax 回归的代价函数
$$
L(\theta)=-\frac{1}{m}\left[\sum_{i=1}^{m} \sum_{j=1}^{k} 1\left\{y_i=j\right\} \log \frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right]\tag{4}
$$
其中，1{·}是示性函数，即1{值为真的表达式}=1，1{值为假的表达式}=0。跟 logistic 函数一样，利用梯度下降法最小化代价函数，下面求解 $\theta$ 的梯度。$L(\theta)$关于 $\theta_{j}$ 的梯度求解为
$$
\begin{aligned} 
\frac{\partial L(\theta)}{\partial \theta_{j}} 
&=-\frac{1}{m} \frac{\partial}{\partial \theta_{j}}\left[\sum_{i=1}^{m} \sum_{j=1}^{k} 1\left\{y_i=j\right\} \log \frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right] \\ 
&=-\frac{1}{m} \frac{\partial}{\partial \theta_{j}}\left[\sum_{i=1}^{m} \sum_{j=1}^{k} 1\left\{y_i=j\right\}\left(\theta_{j}^{T} x_i-\log \sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}\right)\right] \\ 
&=-\frac{1}{m}\left[\sum_{i=1}^{m} 1\left\{y_i=j\right\}\left(x_i-\sum_{j=1}^{k} \frac{e^{\theta_{j}^{T} x_i} \cdot x_i}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right)\right] \\
&=-\frac{1}{m}\left[\sum_{i=1}^{m} x_i1\left\{y_i=j\right\}\left(1-\sum_{j=1}^{k} \frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right)\right] \\
&=-\frac{1}{m}\left[\sum_{i=1}^{m} x_i\left(1\left\{y_i=j\right\}-\sum_{j=1}^{k} 1\left\{y_i=j\right\} \frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right)\right] \\
&=-\frac{1}{m}\left[\sum_{i=1}^{m} x_i\left(1\left\{y_i=j\right\}- \frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right)\right] \\
&=-\frac{1}{m}\left[\sum_{i=1}^{m} x_i\left(1\left\{y_i=j\right\}-p\left(y_i=j | x_i ; \theta\right)\right)\right]
\end{aligned}\tag{5}
$$
感谢 CSDN 博主[2]提供了另外一种求解方法，具体如下
$$
\begin{aligned} 
\frac{\partial L(\theta)}{\partial \theta_{j}} &=-\frac{1}{m}\left[\sum_{i=1}^{m} \frac{\partial}{\partial \theta_{j}}\left(1\left\{y_i=j\right\} \log \frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}+\sum_{c \neq j}^{k} 1\left\{y_i=c\right\} \log \frac{e^{\theta_{c}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right)\right]\\ 
&=-\frac{1}{m}\left[\sum_{i=1}^{m}\left(1\left\{y_i=j\right\}\left(x_i-\frac{e^{\theta_{j}^{T} x_i} \cdot x_i}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right)+\sum_{c \neq j}^{k} 1\left\{y_i=c\right\}\left(-\frac{e^{\theta_{j}^{T} x_i} \cdot x_i}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right)\right)\right]\\ 
&=-\frac{1}{m}\left[\sum_{i=1}^{m} x_i\left(1\left\{y_i=j\right\}\left(1-\frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right)-\sum_{c \neq j}^{k} 1\left\{y_i=c\right\}\frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right)\right] \\
&=-\frac{1}{m}\left[\sum_{i=1}^{m} x_i\left(1\left\{y_i=j\right\}-1\left\{y_i=j\right\} p\left(y_i=j | x_i ; \theta\right)-\sum_{c \neq j}^{k} 1\left\{y_i=c\right\}p\left(y_i=j | x_i ; \theta\right)\right)\right] \\
&=-\frac{1}{m}\left[\sum_{i=1}^{m} x_i\left(1\left\{y_i=j\right\}-\sum_{j=1}^{k} 1\left\{y_i=j\right\} p\left(y_i=j | x_i ; \theta\right)\right)\right] \\
&=-\frac{1}{m}\left[\sum_{i=1}^{m} x_i\left(1\left\{y_i=j\right\}-p\left(y_i=j | x_i ; \theta\right)\right)\right]
\end{aligned}\tag{6}
$$

## 2.2 模型参数特点

softmax 回归有一个不寻常的特点：它有一个“冗余“的参数集。为了便于阐述这一特点，假设我们从参数向量 $\theta_{j}$ 中减去向量 $\psi$ ，那么对于概率函数，我们有
$$
\begin{aligned} 
\left(y_i=j | x_i ; \theta\right)
&=\frac{e^{\left(\theta_{j}- \psi\right)^{T} x_i}}{\sum_{l=1}^{k} e^{\left(\theta_{l}- \psi\right)^{T} x_i}}\\
&=\frac{e^{\theta_{j}^{T} x_i }e^{-\psi^{T} x_i }}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}e^{-\psi^{T} x_i}}\\
&=\frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}
\end{aligned} \tag{7}
$$
换句话说，从参数向量中的每个元素 $\theta_j$ 中减去 $\psi$ 一点也不会影响到假设的类别预测！这表明了 softmax 回归的参数中是有多余的。正式地说， softmax 模型是过参数化的（ overparameterized​ 或参数冗余的），这意味着对任何一个拟合数据的假设而言，多种参数取值有可能得到同样的假设 $h_\theta$，即从输入 $x$ 经过不同的模型参数的假设计算从而得到同样的分类预测结果。  

进一步说，若代价函数 $L(\theta)$ 被某组模型参数 $(\theta_1, \theta_2,\ldots, \theta_k)$ 最小化，那么对任意的 $\psi$ ，代价函数也可以被 $(\theta_1 - \psi, \theta_2 - \psi,\ldots, \theta_k - \psi)$ 最小化。因此， $L(\theta)$ 的最小值时的参数并不唯一。（有趣的是， $L(\theta)$ 仍是凸的，并且在梯度下降中不会遇到局部最优的问题，但是 Hessian​ 矩阵是奇异或不可逆的，这将会导致在牛顿法的直接实现上遇到数值问题。）  

注意到，通过设定 $\psi = \theta_k$ ，总是可以用 $\theta_k - \psi = \vec{0}$代替 $\theta_k$ ，而不会对假设函数有任何影响。因此，可以去掉参数向量 $\theta$ 中的最后一个（或该向量中任意其它任意一个）元素 $\theta_{k}$ ，而不影响假设函数的表达能力。实际上，因参数冗余的特性，与其优化全部的 $k\cdot n$ 个参数 $(\theta_1,\theta_2,\ldots,\theta_k)$ （其中 $\theta_k \in \Re^{n}$），也可令 $\theta_k = \vec{0}$ ，只优化剩余的 $(k-1) \cdot n$ 个参数，算法依然能够正常工作。

## 2.3 正则化

当训练数据不够多的时候，容易出现过拟合现象，拟合系数往往非常大👉[[过拟合原因](https://blog.csdn.net/u012328159/article/details/51089365)]，为此在损失函数后面加上一个正则项，即
$$
L(\theta)=-\frac{1}{m}\left[\sum_{i=1}^{m} \sum_{j=1}^{k} 1\left\{y_i=j\right\} \log \frac{e^{\theta_{j}^{T} x_i}}{\sum_{l=1}^{k} e^{\theta_{l}^{T} x_i}}\right]+\lambda\sum_{i=1}^{k} \sum_{j=1}^{n}\theta_{ij}^{2} \tag{8}
$$
那么新的损失函数的梯度为
$$
\frac{\partial L(\theta)}{\partial \theta_{j}} =-\frac{1}{m}\left[\sum_{i=1}^{m} x_i\left(1\left\{y_i=j\right\}-p\left(y_i=j | x_i ; \theta\right)\right)\right]+\lambda\theta_j \tag{9}
$$
⚠️**注意**：上式中的 $\theta_j$ 中的 $\theta_{j0}$ 不应该被惩罚，因为他是一个常数项，所以在实际使用的时候仅仅需要对 $\theta_{j1},\theta_{j2},\dots,\theta_{jn}$ 进行惩罚即可，这个会在后面的 python 代码中提到😃。

## 2.4 softmax 与 logistic 回归的关系

文章开头说过，softmax 回归是 logistic 回归的一般形式，logistic 回归是 softmax 回归在 $k=2$ 时的特殊形式，下面通过公式推导来看下当 $k=2$ 时 softmax 回归是如何退化成 logistic 回归。

当 $k=2$ 时，softmax 回归的假设函数为
$$
h_{\theta}\left(x_i\right)=\left[\begin{array}{c}{p\left(y_i=1 | x_i ; \theta\right)} \\ {p\left(y_i=2 | x_i ; \theta\right)} \end{array}\right]=\frac{1}{e^{\theta_{1}^{T} x_i}+e^{\theta_{2}^{T} x_i}}\left[\begin{array}{c}{e^{\theta_{1}^{T} x_i}} \\ {e^{\theta_{2}^{T} x_i}} \end{array}\right]\tag{10}
$$
前面说过 softmax 回归的参数具有冗余性，从参数向量 $\theta_1,\theta_2$ 中减去向量 $\theta_1$完全不影响结果。现在我们令 $\theta'=\theta_2-\theta_1$，并且两个参数向量都减去 $\theta_1$，则有
$$
\begin{aligned} 
h_{\theta}(x_i) &=\frac{1}{e^{\vec{0}^{T} x_i}+e^{\left(\theta_{2}-\theta_{1}\right)^{T} x_i}}\left[\begin{array}{c}{e^{\vec{0}^{T} x_i}} \\ {e^{\left(\theta_{2}-\theta_{1}\right)^{T} x_i}}\end{array}\right] \\ 
&=\left[\begin{array}{c}{\frac{1}{1+e^{\left(\theta_{2}-\theta_{1}\right)^{T} x_i}}} \\ {\frac{e^{\left(\theta_{2}-\theta_{1}\right)^{T} x_i}}{1+e^{\left(\theta_{2}-\theta_{1}\right)^{T} x_i}}}\end{array}\right] \\
&=\left[\begin{array}{c}{\frac{1}{1+e^{\left(\theta_{2}-\theta_{1}\right)^{T} x_i}}} \\ 1-{\frac{1}{1+e^{\left(\theta_{2}-\theta_{1}\right)^{T} x_i}}}\end{array}\right] \\
&=\left[\begin{array}{c}{\frac{1}{1+e^{\left(\theta'\right)^{T} x_i}}} \\ 1-{\frac{1}{1+e^{\left(\theta'\right)^{T} x_i}}}\end{array}\right] \\
\end{aligned}\tag{11}
$$
这样就化成了 logistic 回归。

# 三、实现

## 3.1 python 手动实现

这里的数据使用的是 sklearn 的算法包生成的随机数据，其中，训练数据为 3750×2 的数据，测试数据为 1250×2 的数据，生成代码如下

```python
def gen_dataset():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    np.random.seed(13)
    X, y = make_blobs(centers=4, n_samples = 5000)
    # 绘制数据分布
    plt.figure(figsize=(6,4))
    plt.scatter(X[:,0], X[:,1],c=y)
    plt.title("Dataset")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()

    # 重塑目标以获得具有 (n_samples, 1)形状的列向量
    y = y.reshape((-1,1))
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_dataset = np.append(X_train,y_train, axis = 1)
    test_dataset = np.append(X_test,y_test, axis = 1)
    np.savetxt("train_dataset.txt", train_dataset, fmt="%.4f %.4f %d")
    np.savetxt("test_dataset.txt", test_dataset, fmt="%.4f %.4f %d")
```

数据分布情况如下图所示

<img src="https://tva1.sinaimg.cn/large/006tNbRwly1ga0om326ljj30xc0m8q54.jpg" alt="dataset" style="zoom:50%;" />

softmax 算法的核心部分就是求解梯度矩阵，我们设输入数据为 $X=\{x_1,x_2,\ldots,x_m\}$，这是一个 $m×n$ 的矩阵，输出类别为 $y=\{y_1,y_2,\ldots,y_m\}$，其中 $y_i$ 是一个 $1×k$ 的one-hot 矩阵，$k$ 表示类别个数，那么 $y$ 其实是一个 $m×k$ 的矩阵，输入数据对应的概率为 $P=\{p_1,p_2,\ldots,p_m\}$， 同样的这也是一个 $m×k$ 的矩阵。那么根据公式(9)，可以知道 $\theta_j$ 的梯度为 
$$
\frac{\partial L(\theta)}{\partial \theta_{j}} =-\frac{1}{m}\left(y_i-P_i\right)^TX+\lambda\theta_j \tag{12}
$$
由此可以推导出 $\theta$ 的参数矩阵为
$$
\frac{\partial L(\theta)}{\partial \theta} =-\frac{1}{m}\left(y-P\right)^TX+\lambda\theta \tag{13}
$$
注意到这里也考虑了 $\theta_j$ 的第 0 项 ，所以在写代码的时候需要把 $\theta$ 的第 0 列的惩罚项减去。

softmax 回归的代码如下

```python
def load_dataset(file_path):
    dataMat = []
    labelMat = []
    fr = open(file_path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def train(data_arr, label_arr, n_class, iters = 1000, alpha = 0.1, lam = 0.01):
    '''
    @description: softmax 训练函数
    @param {type} 
    @return: theta 参数
    '''    
    n_samples, n_features = data_arr.shape
    n_classes = n_class
    # 随机初始化权重矩阵
    weights = np.random.rand(n_class, n_features)
    # 定义损失结果
    all_loss = list()
    # 计算 one-hot 矩阵
    y_one_hot = one_hot(label_arr, n_samples, n_classes)
    for i in range(iters):
        # 计算 m * k 的分数矩阵
        scores = np.dot(data_arr, weights.T)
        # 计算 softmax 的值
        probs = softmax(scores)
        # 计算损失函数值
        loss = - (1.0 / n_samples) * np.sum(y_one_hot * np.log(probs))
        all_loss.append(loss)
        # 求解梯度
        dw = -(1.0 / n_samples) * np.dot((y_one_hot - probs).T, data_arr) + lam * weights
        dw[:,0] = dw[:,0] - lam * weights[:,0]
        # 更新权重矩阵
        weights  = weights - alpha * dw
    return weights, all_loss
        

def softmax(scores):
    # 计算总和
    sum_exp = np.sum(np.exp(scores), axis = 1,keepdims = True)
    softmax = np.exp(scores) / sum_exp
    return softmax


def one_hot(label_arr, n_samples, n_classes):
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), label_arr.T] = 1
    return one_hot


def predict(test_dataset, label_arr, weights):
    scores = np.dot(test_dataset, weights.T)
    probs = softmax(scores)
    return np.argmax(probs, axis=1).reshape((-1,1))


if __name__ == "__main__":
    #gen_dataset()
    data_arr, label_arr = load_dataset('train_dataset.txt')
    data_arr = np.array(data_arr)
    label_arr = np.array(label_arr).reshape((-1,1))
    weights, all_loss = train(data_arr, label_arr, n_class = 4)

    # 计算预测的准确率
    test_data_arr, test_label_arr = load_dataset('test_dataset.txt')
    test_data_arr = np.array(test_data_arr)
    test_label_arr = np.array(test_label_arr).reshape((-1,1))
    n_test_samples = test_data_arr.shape[0]
    y_predict = predict(test_data_arr, test_label_arr, weights)
    accuray = np.sum(y_predict == test_label_arr) / n_test_samples
    print(accuray)

    # 绘制损失函数
    fig = plt.figure(figsize=(8,5))
    plt.plot(np.arange(1000), all_loss)
    plt.title("Development of loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()
```

函数输出的测试数据准确率为

```
0.9952
```

程序中记录了每个循环的损失函数，其变化曲线如下图所示。

<img src="https://tva1.sinaimg.cn/large/006tNbRwly1ga0p55zfokj318g0rsta0.jpg" alt="loss function" style="zoom:50%;" />

## 3.2 sklearn 算法包实现

`sklearn`的实现比较简单，与 logistic 回归的代码类似。

```python
def softmax_lib():
    data_arr, label_arr = load_dataset('train_dataset.txt')
    from sklearn import linear_model
    model_softmax_regression = linear_model.LogisticRegression(solver='lbfgs',multi_class="multinomial",max_iter=10)
    model_softmax_regression.fit(data_arr, label_arr)
    test_data_arr, test_label_arr = load_dataset('test_dataset.txt')
    y_predict = model_softmax_regression.predict(test_data_arr)
    accurcy = np.sum(y_predict == test_label_arr) / len(test_data_arr)
    print(accurcy)
```
输出结果为
```
0.9848
```

本文的完整代码和数据去👉[[我的 github]](https://github.com/HuStanding/nlp-exercise/blob/master/softmax/softmax.py)查看

# 四、参考

[1] https://zhuanlan.zhihu.com/p/34520042

[2] https://blog.csdn.net/u012328159/article/details/72155874

[3] https://zhuanlan.zhihu.com/p/56139075

[4] [https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0](https://zh.wikipedia.org/wiki/Softmax函数)

[5] http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/

