# 一、什么是聚类

## 1.1 聚类的定义

`聚类(Clustering)`是按照某个特定标准(如距离)把一个数据集分割成不同的类或簇，使得**同一个簇内的数据对象的相似性尽可能大，同时不在同一个簇中的数据对象的差异性也尽可能地大**。也即聚类后同一类的数据尽可能聚集到一起，不同类数据尽量分离。

## 1.2 聚类和分类的区别

+ `聚类(Clustering)`：是指把相似的数据划分到一起，具体划分的时候并不关心这一类的标签，目标就是把相似的数据聚合到一起，聚类是一种`无监督学习(Unsupervised Learning)`方法。
+ `分类(Classification)`：是把不同的数据划分开，其过程是通过训练数据集获得一个分类器，再通过分类器去预测未知数据，分类是一种`监督学习(Supervised Learning)`方法。

## 1.3 聚类的一般过程

1. 数据准备：特征标准化和降维
2. 特征选择：从最初的特征中选择最有效的特征，并将其存储在向量中
3. 特征提取：通过对选择的特征进行转换形成新的突出特征
4. 聚类：基于某种距离函数进行相似度度量，获取簇
5. 聚类结果评估：分析聚类结果，如`距离误差和(SSE)`等

## 1.4 数据对象间的相似度度量

对于数值型数据，可以使用下表中的相似度度量方法。

| <div style="width:120px">**相似度度量准则**</div> | **相似度度量函数** |
| :------------: | :------------: |
| Euclidean 距离 | $d( x, y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$ |
| Manhattan 距离 | $d( x, y)=\sum_{i=1}^{n}\left \|x_i-y_i \right\|$ |
| Chebyshev 距离 | $d( x, y)=\max_{i=1,2,…,n}^{n}\left\|x_i-y_i \right\|$ |
| Minkowski 距离 | $d( x, y)=[\sum_{i=1}^{n}(x_i-y_i)^p]^ {\frac{1}{p}}$ |

`Minkowski `距离就是$ Lp $范数（$p≥1$)，而 `Manhattan` 距离、`Euclidean `距离、`Chebyshev `距离分别对应 $p=1,2,∞ $时的情形。

## 1.5 cluster之间的相似度度量

除了需要衡量对象之间的距离之外，有些聚类算法（如层次聚类）还需要衡量`cluster`之间的距离 ，假设$ C_i $和$ C_j$ 为两个 `cluster`，则前四种方法定义的 $C_i $和 $C_j$ 之间的距离如下表所示：

| **相似度度量准则** | **相似度度量函数** |
| :------------: | :------------: |
| Single-link | $D(C_i,C_j)= \min_{x\subseteq C_i, y\subseteq C_j}d( x, y)$ |
| Complete-link | $D(C_i,C_j)= \max_{x\subseteq C_i, y\subseteq C_j}d( x, y)$ |
| UPGMA | $D(C_i,C_j)= \frac{1}{\left \| C_i\right \|\left \| C_j\right \|}\sum_{x\subseteq C_i, y\subseteq C_j}d( x, y)$ |
| WPGMA | - |

+ `Single-link`定义两个`cluster`之间的距离为两个`cluster`之间距离最近的两个点之间的距离，这种方法会在聚类的过程中产生`链式效应`，即有可能会出现非常大的`cluster`
+ `Complete-link`定义的是两个`cluster`之间的距离为两个``cluster`之间距离最远的两个点之间的距离，这种方法可以避免`链式效应`,对异常样本点（不符合数据集的整体分布的噪声点）却非常敏感，容易产生不合理的聚类
+ `UPGMA`正好是`Single-link`和`Complete-link`方法的折中，他定义两个`cluster`之间的距离为两个`cluster`之间所有点距离的平均值
+ 最后一种`WPGMA`方法计算的是两个 `cluster` 之间两个对象之间的距离的加权平均值，加权的目的是为了使两个 `cluster` 对距离的计算的影响在同一层次上，而不受 `cluster` 大小的影响，具体公式和采用的权重方案有关。

# 二、数据聚类方法

数据聚类方法主要可以分为`划分式聚类方法(Partition-based Methods)`、`基于密度的聚类方法(Density-based methods)`、`层次化聚类方法(Hierarchical Methods)`等。

![聚类方法](https://tva1.sinaimg.cn/large/006y8mN6ly1g8t96s2dt3j30ze0qatbk.jpg)

## 2.1 划分式聚类方法

 划分式聚类方法需要事先指定簇类的数目或者聚类中心，通过反复迭代，直至最后达到<font color=red>"簇内的点足够近，簇间的点足够远"</font>的目标。经典的划分式聚类方法有`k-means`及其变体`k-means++`、`bi-kmeans`、`kernel k-means`等。

### 2.1.2 k-means算法

经典的`k-means`算法的流程如下：

>1. 创建$k$个点作为初始质心(通常是随机选择)
>2. 当任意一个点的簇分配结果发生改变时
>    1. 对数据集中的每个数据点
>        1. 对每个质心
>            1. 计算质心与数据点之间的距离
>        2. 将数据点分配到距其最近的簇
>    2. 对每个簇，计算簇中所有点的均值并将均值作为质心

经典`k-means`[源代码](https://github.com/HuStanding/nlp-exercise/blob/master/cluster/kmeans/kmeans.py)，下左图是原始数据集，通过观察发现大致可以分为4类，所以取$k=4$，测试数据效果如下右图所示。
![kmeans](https://tva1.sinaimg.cn/large/006y8mN6ly1g8uevrqhbtj31uo0qowh4.jpg)

![k-means聚类过程](https://tva1.sinaimg.cn/large/006y8mN6ly1g8vij16f5tg30hs0dc43b.gif)

看起来很顺利，但事情并非如此，我们考虑`k-means`算法中最核心的部分，假设$x_i(i=1,2,…,n)$是数据点，$\mu_j(j=1,2,…,k)$是初始化的数据中心，那么我们的目标函数可以写成
$$
\min\sum_{i=1}^{n} \min \limits_{j=1,2,...,k}\left |\left |  x_i -\mu_j\right | \right |^2
$$
这个函数是**非凸优化函数**，会收敛于局部最优解，可以参考[证明过程](https://math.stackexchange.com/questions/463453/how-to-see-that-k-means-objective-is-convex)。举个🌰，$\mu_1=\left [ 1,1\right ] ,\mu_2=\left [ -1,-1\right ]$，则
$$
z=\min \limits_{j=1,2}\left |\left |  x_i -\mu_j\right | \right |^2
$$
该函数的曲线如下图所示

![局部最优](https://tva1.sinaimg.cn/large/006y8mN6ly1g8uewpxhkgj30rb0lpqe0.jpg)

可以发现该函数有两个局部最优点，当时初始质心点取值不同的时候，最终的聚类效果也不一样，接下来我们看一个具体的实例。

![划分错误](https://tva1.sinaimg.cn/large/006y8mN6ly1g8ueyh4wcej31uo0qo0uu.jpg)

在这个例子当中，下方的数据应该归为一类，而上方的数据应该归为两类，这是由于初始质心点选取的不合理造成的误分。而$k$值的选取对结果的影响也非常大，同样取上图中数据集，取$k=2,3,4$，可以得到下面的聚类结果：

![k值不同](https://tva1.sinaimg.cn/large/006y8mN6ly1g8uez8ra7cj31uo0hswgu.jpg)

一般来说，经典`k-means`算法有以下几个特点：

1. 需要提前确定$k$值
2. 对初始质心点敏感
3. 对异常数据敏感

### 2.1.2 k-means++算法

`k-means++`是针对`k-means`中初始质心点选取的优化算法。该算法的流程和`k-means`类似，改变的地方只有初始质心的选取，该部分的算法流程如下

> 1. 随机选取一个数据点作为初始的聚类中心
> 2. 当聚类中心数量小于$k$
>     1. 计算每个数据点与当前已有聚类中心的最短距离，用$D(x)$表示，这个值越大，表示被选取为下一个聚类中心的概率越大，最后使用轮盘法选取下一个聚类中心

`k-means++`[源代码](https://github.com/HuStanding/nlp-exercise/blob/master/cluster/kmeans/kmeans%2B%2B.py)，使用`k-means++`对上述数据做聚类处理，得到的结果如下

![k-means++](https://tva1.sinaimg.cn/large/006y8mN6ly1g8uf1hnybsj31uo0qomzc.jpg)

### 2.1.3 bi-kmeans算法

一种度量聚类效果的指标是`SSE(Sum of Squared Error)`，他表示聚类后的簇离该簇的聚类中心的平方和，`SSE`越小，表示聚类效果越好。 `bi-kmeans`是针对`kmeans`算法会陷入局部最优的缺陷进行的改进算法。该算法基于SSE最小化的原理，首先将所有的数据点视为一个簇，然后将该簇一分为二，之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否能最大程度的降低`SSE`的值。

该算法的流程如下：

> 1. 将所有点视为一个簇
> 2. 当簇的个数小于$k$时
>     1. 对每一个簇
>         1. 计算总误差
>         2. 在给定的簇上面进行`k-means`聚类($k=2$)
>         3. 计算将该簇一分为二之后的总误差
>     2. 选取使得误差最小的那个簇进行划分操作

`bi-kmeans`算法[源代码](https://github.com/HuStanding/nlp-exercise/blob/master/cluster/kmeans/bi_kmeans.py)，利用`bi-kmeans`算法处理上节中的数据得到的结果如下图所示。

![bi-kmeans](https://tva1.sinaimg.cn/large/006y8mN6ly1g8uf25kdmxj31uo0qowgo.jpg)

这是一个全局最优的方法，所以每次计算出来的`SSE`值肯定也是一样的，我们和前面的`k-means`、`k-means++`比较一下计算出来的`SSE`值

| 序号 | k-means | k-means++ | bi-kmeans |
|------|---------|-----------|-----------|
| 1    | 2112    | 120       | 106       |
| 2    | 388     | 125       | 106       |
| 3    | 824     | 127       | 106       |
| agv  | 1108    | 124       | 106       |

可以看到，`k-means`每次计算出来的`SSE`都较大且不太稳定，`k-means++`计算出来的`SSE`较稳定并且数值较小，而`bi-kmeans`每次计算出来的`SSE`都一样(因为是全局最优解)并且计算的`SSE`都较小，说明聚类的效果也最好。

## 2.2 基于密度的方法

`k-means`算法对于凸性数据具有良好的效果，能够根据距离来讲数据分为球状类的簇，但对于非凸形状的数据点，就无能为力了，当`k-means`算法在环形数据的聚类时，我们看看会发生什么情况。

![k-means的局限性](https://tva1.sinaimg.cn/large/006y8mN6ly1g8w8umxp7yj31uo0qo476.jpg)

从上图可以看到，`kmeans`聚类产生了错误的结果，这个时候就需要用到基于密度的聚类方法了，该方法需要定义两个参数$\varepsilon$和$M$，分别表示密度的邻域半径和邻域密度阈值。`DBSCAN`就是其中的典型。

### 2.2.1 DBSCAN算法

首先介绍几个概念，考虑集合$X=\left \{x^{(1)},x^{(2)},...,x^{(n)}\right \}$，$\varepsilon$表示定义密度的邻域半径，设聚类的邻域密度阈值为$M$，有以下定义：

+ **$\varepsilon$邻域($\varepsilon$-neighborhood）**

$$
N_{\varepsilon }(x)=\left \{y\in  X|d(x, y) < \varepsilon \right \}
$$
+ **密度(desity)**
$x$的密度为
$$
\rho (x)=\left | N_{\varepsilon }(x)\right |
$$

+ **核心点(core-point)**

设$x\in  X$，若$\rho (x) \geq M$，则称$x$为$X$的核心点，记$X$中所有核心点构成的集合为$X_c$，记所有非核心点构成的集合为$X_{nc}$。

+ **边界点(border-point)**

若$x\in  X_{nc}$，且$\exists y\in  X$，满足
$$
y\in  N_{\varepsilon }(x) \cap X_c
$$
即$x$的$\varepsilon$邻域中存在核心点，则称$x$为$X$的边界点，记$X$中所有的边界点构成的集合为$X_{bd}$。

此外，边界点也可以这么定义：若$x\in  X_{nc}$，且$x$落在某个核心点的$\varepsilon$邻域内，则称$x$为$X$的一个边界点，一个边界点可能同时落入一个或多个核心点的$\varepsilon$邻域。

+ **噪声点(noise-point)**

若$x$满足
$$
x\in  X,x \notin X_{c}且 x\notin X_{bd}
$$
则称$x$为噪声点。

如下图所示，设$M=3$，则A为核心点，B、C是边界点，而N是噪声点。

![形象化解释](https://tva1.sinaimg.cn/large/006y8mN6ly1g8p89zkw6dj30ew09640g.jpg)

该算法的流程如下：

> 1. 标记所有对象为unvisited
> 2. 当有标记对象时
>     1. 随机选取一个unvisited对象$p$
>     2. 标记$p$为visited
>     3. 如果$p$的$\varepsilon $邻域内至少有$M$个对象，则
>         1. 创建一个新的簇$C$，并把$p$放入$C$中
>         2. 设$N$是$p$的$\varepsilon $邻域内的集合，对$N$中的每个点$p'$
>             1. 如果点$p'$是unvisited
>                 1. 标记$p'$为visited
>                 2. 如果$p'$的$\varepsilon $邻域至少有$M$个对象，则把这些点添加到$N$
>                 3. 如果$p'$还不是任何簇的成员，则把$p'$添加到$C$
>         3. 保存$C$
>     4. 否则标记$p$为噪声

构建$\varepsilon$邻域的过程可以使用`kd-tree`进行优化，循环过程可以使用`Numba、Cython、C`进行[优化](https://blog.csdn.net/weixin_38169413/article/details/102729497)，`DBSCAN`的[源代码](https://github.com/HuStanding/nlp-exercise/blob/master/cluster/dbscan/dbscan.py)，使用该节一开始提到的数据集，聚类效果如下

![DBSCAN](https://tva1.sinaimg.cn/large/006y8mN6ly1g8w671fsgyj31uo0qo7b0.jpg)

聚类的过程示意图

![DBSCAN 处理过程](https://tva1.sinaimg.cn/large/006y8mN6ly1g8w7t41boyg30hs0dcb2b.gif)

当设置不同的$\varepsilon $时，会产生不同的结果，如下图所示

![不同$\varepsilon$下的聚类效果](https://tva1.sinaimg.cn/large/006y8mN6ly1g8uenpb7ycj30xc0qodmd.jpg)

当设置不同的$M$时，会产生不同的结果，如下图所示

![不同$M$下的聚类效果](https://tva1.sinaimg.cn/large/006y8mN6ly1g8uepldesqj30xc0qo0yo.jpg)

一般来说，`DBSCAN`算法有以下几个特点：

1. 需要提前确定$\varepsilon $和$M$值
2. 不需要提前设置聚类的个数
3. 对初值选取敏感，对噪声不敏感
4. 对密度不均的数据聚合效果不好

### 2.2.2 OPTICS算法

在`DBSCAN`算法中，使用了统一的$\varepsilon$值，当数据密度不均匀的时候，如果设置了较小的$\varepsilon$值，则较稀疏的`cluster`中的节点密度会小于$M$，会被认为是边界点而不被用于进一步的扩展；如果设置了较大的$\varepsilon$值，则密度较大且离的比较近的`cluster`容易被划分为同一个`cluster`，如下图所示。

![密度不均的一种情况](https://tva1.sinaimg.cn/large/006y8mN6ly1g8w8yv1ronj30cg0a53z5.jpg)

+ 如果设置的$\varepsilon$较大，将会获得A,B,C这3个`cluster`
+ 如果设置的$\varepsilon$较小，将会只获得C1、C2、C3这3个`cluster`

对于密度不均的数据选取一个合适的$\varepsilon$是很困难的，对于高维数据，由于**维度灾难(Curse of dimensionality)**,$\varepsilon$的选取将变得更加困难。

怎样解决`DBSCAN`遗留下的问题呢？

> <font >The basic idea to overcome these problems is to run an algorithm which produces a special order of the database with respect to its density-based clustering structure containing the information about every clustering level of the data set (up to a "generating distance" $\varepsilon$), and is very easy to analyze.

即能够提出一种算法，使得基于密度的聚类结构能够呈现出一种特殊的顺序，该顺序所对应的聚类结构包含了每个层级的聚类的信息，并且便于分析。

`OPTICS(Ordering Points To Identify the Clustering Structure, OPTICS)`实际上是`DBSCAN`算法的一种有效扩展，主要解决对输入参数敏感的问题。即选取有限个邻域参数$\varepsilon _i( 0 \leq\varepsilon_{i} \leq \varepsilon)$  进行聚类，这样就能得到不同邻域参数下的聚类结果。

在介绍`OPTICS`算法之前，再扩展几个概念。

+ **核心距离(core-distance)**

样本$x∈X$，对于给定的$\varepsilon$和$M$，使得$x$成为核心点的最小邻域半径称为$x$的核心距离，其数学表达如下
$$
cd(x)=\left\{\begin{matrix}
UNDEFINED, \left | N_{\varepsilon }(x)\right |< M\\ 
d(x,N_{\varepsilon }^{M}(x)), \left | N_{\varepsilon }(x)\right | \geqslant  M
\end{matrix}\right.
$$


其中，$N_{\varepsilon }^{i}(x)$表示在集合$N_{\varepsilon }(x)$中与节点$x$第$i$近邻的节点，如$N_{\varepsilon }^{1}(x)$表示$N_{\varepsilon }(x)$中与$x$最近的节点，如果$x$为核心点，则必然会有$cd(x) \leq\varepsilon$。

+ **可达距离(reachability-distance)**

设$x,y∈X$，对于给定的参数$\varepsilon$和$M$，$y$关于$x$的可达距离定义为
$$
rd(y,x)=\left\{\begin{matrix}
UNDEFINED, \left | N_{\varepsilon }(x)\right |< M\\ 
\max{\{cd(x),d(x,y)\}}, \left|  N_{\varepsilon }(x)\right | \geqslant  M
\end{matrix}\right.
$$
特别地，当$x$为核心点时，可以按照下式来理解$rd(x,y)$的含义
$$
rd(x,y)=\min\{\eta:y \in N_{\eta}(x) 且 \left|N_{\eta}(x)\right| \geq M\}
$$
即$rd(x,y)$表示使得"$x$为核心点"且"$y$从$x$直接密度可达"同时成立的最小邻域半径。

> 可达距离的意义在于衡量$y$所在的密度，密度越大，他从相邻节点直接密度可达的距离越小，如果聚类时想要朝着数据尽量稠密的空间进行扩张，那么可达距离最小是最佳的选择。

举个🌰，下图中假设$M=3$，半径是$ε$。那么$P$点的核心距离是$d(1,P)$，点2的可达距离是$d(1,P)$，点3的可达距离也是$d(1,P)$，点4的可达距离则是$d(4,P)$的距离。

![核心距离与可达距离](https://tva1.sinaimg.cn/large/006y8mN6ly1g8t8qlyvo3j30ic09qmyq.jpg)

`OPTICS`[源代码](https://github.com/HuStanding/nlp-exercise/blob/master/cluster/dbscan/optics.py)，算法流程如下：

> 1. 标记所有对象为unvisited，初始化order_list为空
> 2. 当有标记对象时
>     1. 随机选取一个unvisited对象$i$
>     2. 标记$i$为visited，插入结果序列order_list中
>     3. 如果$i$的$\varepsilon$邻域内至少有$M$个对象，则
>         1. 初始化seed_list种子列表
>         2. 调用insert_list()，将邻域对象中未被访问的节点按照可达距离插入队列seeld_list中
>         3. 当seed_list列表不为空
>             1. 按照可达距离升序取出seed_list中第一个元素$j$
>             2. 标记$j$为visited，插入结果序列order_list中
>             3. 如果$j$的$\varepsilon $邻域内至少有$M$个对象，则
>                 1. 调用insert_list()，将邻域对象中未被访问的节点按照可达距离插入队列seeld_list中

算法中有一个很重要的insert_list()函数，这个函数如下：

> 1. 对$i$中所有的邻域点$k$
> 2. 如果$k$未被访问过
>     1. 计算$rd(k,i)$
>     2. 如果$r_k=UNDEFINED$
>         1. $r_k=rd(k,i)$
>         2. 将节点$k$按照可达距离插入seed_list中
>     3. 否则
>         1. 如果 $rd(k,i)<r_k$
>         2. 更新$r_k$的值，并按照可达距离重新插入seed_list中

`OPTICS`算法输出序列的过程：

![OPTICS处理过程](https://tva1.sinaimg.cn/large/006y8mN6ly1g8wf2xw78jg30hs0dctxb.gif)

该算法最终获取知识是一个**输出序列**，该序列按照密度不同将相近密度的点聚合在一起，而不是输出该点所属的具体类别，如果要获取该点所属的类型，需要再设置一个参数$\varepsilon'(\varepsilon' \leq \varepsilon)$提取出具体的类别。这里我们举一个例子就知道是怎么回事了。

随机生成三组密度不均的数据，我们使用`DBSCAN`和`OPTICS`来看一下效果。

![DBSCAN划分不均匀数据](https://tva1.sinaimg.cn/large/006y8mN6ly1g8v0urgit2j31uo0qo78u.jpg)

![OPTICS输出序列和分类结果](https://tva1.sinaimg.cn/large/006y8mN6ly1g8vdrmq2e6j31uo0qodkh.jpg)

可见，`OPTICS`第一步生成的输出序列较好的保留了各个不同密度的簇的特征，根据输出序列的可达距离图，再设定一个合理的$\varepsilon'$，便可以获得较好的聚类效果。

## 2.3 层次化聚类方法

前面介绍的几种算法确实可以在较小的复杂度内获取较好的结果，但是这几种算法却存在一个`链式效应`的现象，比如：A与B相似，B与C相似，那么在聚类的时候便会将A、B、C聚合到一起，但是如果A与C不相似，就会造成聚类误差，严重的时候这个误差可以一直传递下去。为了降低`链式效应`，这时候层次聚类就该发挥作用了。

![链式效应](https://tva1.sinaimg.cn/large/006y8mN6ly1g8vduffj04j30c505paa3.jpg)

**层次聚类算法 (hierarchical clustering)** 将数据集划分为一层一层的 `clusters`，后面一层生成的 `clusters` 基于前面一层的结果。层次聚类算法一般分为两类：

- **Agglomerative 层次聚类**：又称自底向上（bottom-up）的层次聚类，每一个对象最开始都是一个 `cluster`，每次按一定的准则将最相近的两个 `cluster` 合并生成一个新的 `cluster`，如此往复，直至最终所有的对象都属于一个 `cluster`。这里主要关注此类算法。
- **Divisive 层次聚类**： 又称自顶向下（top-down）的层次聚类，最开始所有的对象均属于一个 `cluster`，每次按一定的准则将某个 `cluster` 划分为多个 `cluster`，如此往复，直至每个对象均是一个 `cluster`。

![层次聚类过程](https://tva1.sinaimg.cn/large/006y8mN6ly1g8ipc9dng5j310k0pgwgp.jpg)

另外，需指出的是，层次聚类算法是一种贪心算法（greedy algorithm），因其每一次合并或划分都是基于某种局部最优的选择。

### 2.3.1 Agglomerative算法

给定数据集 $X=\left \{x^{(1)},x^{(2)},...,x^{(n)}\right \}$，`Agglomerative `层次聚类最简单的实现方法分为以下几步：

> 1. 初始时每个样本为一个 `cluster`，计算距离矩阵 $D$，其中元素$D_{ij}$为样本点$D_i$和 $D_j$ 之间的距离；
> 2. 遍历距离矩阵 $D$，找出其中的最小距离（对角线上的除外），并由此得到拥有最小距离的两个 `cluster` 的编号，将这两个 `cluster` 合并为一个新的 `cluster` 并依据 `cluster`距离度量方法更新距离矩阵$D$（删除这两个 `cluster` 对应的行和列，并把由新 `cluster` 所算出来的距离向量插入 $D$中），存储本次合并的相关信息；
> 3. 重复 2 的过程，直至最终只剩下一个 `cluster` 。
> 

`Agglomerative`算法[源代码](https://github.com/HuStanding/nlp-exercise/blob/master/cluster/hierarchical/hierarchical.py)，可以看到，该 算法的时间复杂度为 $O(n^3)$ （由于每次合并两个 `cluster` 时都要遍历大小为 $O(n^2) $的距离矩阵来搜索最小距离，而这样的操作需要进行 $n−1$ 次），空间复杂度为$O(n^2) $ （由于要存储距离矩阵）。

![不同类簇度量方法下的层次聚类效果](https://tva1.sinaimg.cn/large/006y8mN6ly1g8syxs9refj30zk0qo79g.jpg)

上图中分别使用了层次聚类中4个不同的`cluster`度量方法，可以看到，使用`single-link`确实会造成一定的链式效应，而使用`complete-link`则完全不会产生这种现象，使用`average-link`和`ward-link`则介于两者之间。

## 2.4 聚类方法比较

| <div style="width:100px">算法类型</div>| <div style="width:80px">适合的数据类型</div> | <div style="width:50px">抗噪点性能</div> | <div style="width:80px">聚类形状</div> | <div style="width:40px">算法效率</div> |
| ------------- | -------------- | ---------- | -------- | -------- |
| kmeans        | 混合型         | 较差       | 球形     | 很高     |
| k\-means\+\+  | 混合型         | 一般       | 球形     | 较高     |
| bi\-kmeans    | 混合型         | 一般       | 球形     | 较高     |
| DBSCAN        | 数值型         | 较好       | 任意形状 | 一般     |
| OPTICS        | 数值型         | 较好       | 任意形状 | 一般     |
| Agglomerative | 混合型         | 较好       | 任意形状 | 较差     |

# 三、参考文献

[1] 李航.统计学习方法

[2] Peter Harrington.Machine Learning in Action/李锐.机器学习实战

[3] https://www.zhihu.com/question/34554321

[4] [T. Soni Madhulatha.AN OVERVIEW ON CLUSTERING METHODS](https://arxiv.org/pdf/1205.1117.pdf)

[5] https://zhuanlan.zhihu.com/p/32375430

[6] [http://heathcliff.me/聚类分析（一）：层次聚类算法](http://heathcliff.me/聚类分析（一）：层次聚类算法/)

[7] https://www.cnblogs.com/tiaozistudy/p/dbscan_algorithm.html

[8] https://blog.csdn.net/itplus/article/details/10089323

[9] [Mihael Ankerst.OPTICS: ordering points to identify the clustering structure](http://www2.denizyuret.com/ref/ankerst/OPTICS.pdf)