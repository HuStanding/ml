# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-14 20:51:30
# @Last Modified by:   huzhu
# @Last Modified time: 2019-10-29 18:36:39

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from scipy.spatial import KDTree

class visitlist:
    """
        visitlist类用于记录访问列表
        unvisitedlist记录未访问过的点
        visitedlist记录已访问过的点
        unvisitednum记录访问过的点数量
    """
    def __init__(self, count=0):
        self.unvisitedlist=[i for i in range(count)]
        self.visitedlist=list()
        self.unvisitednum=count

    def visit(self, pointId):
        self.visitedlist.append(pointId)
        self.unvisitedlist.remove(pointId)
        self.unvisitednum -= 1


def dist(a, b):
    """
    @brief      计算a,b两个元组的欧几里得距离
    @param      a     
    @param      b     
    @return     距离
    """
    return math.sqrt(np.power(a-b, 2).sum())

    
def dbscan_simple(dataSet, eps, minPts):
    """
    @brief      简易DBScan算法
    @param      dataSet  输入数据集，numpy格式
    @param      eps      最短距离
    @param      minPts   最小簇点数
    @return     分类标签
    """
    nPoints = dataSet.shape[0]
    vPoints = visitlist(count=nPoints)
    # 初始化簇标记列表C,簇标记为 k
    k = -1
    C = [-1 for i in range(nPoints)]
    while(vPoints.unvisitednum > 0):
        p = random.choice(vPoints.unvisitedlist)
        vPoints.visit(p)
        N = [i for i in range(nPoints) if dist(dataSet[i], dataSet[p])<= eps]
        if  len(N) >= minPts:
            k += 1
            C[p]=k
            for p1 in N:
                if p1 in vPoints.unvisitedlist:
                    vPoints.visit(p1)
                    M=[i for i in range(nPoints) if dist(dataSet[i], \
                        dataSet[p1]) <= eps]
                    if len(M) >= minPts:
                        for i in M:
                            if i not in N:
                                N.append(i)
                    if  C[p1] == -1:
                        C[p1]= k
        else:
            C[p]=-1
    return C


def dbscan(dataSet, eps, minPts):
    """
    @brief      基于kd-tree的DBScan算法
    @param      dataSet  输入数据集，numpy格式
    @param      eps      最短距离
    @param      minPts   最小簇点数
    @return     分类标签
    """
    nPoints = dataSet.shape[0]
    vPoints = visitlist(count=nPoints)
    # 初始化簇标记列表C，簇标记为 k
    k = -1
    C = [-1 for i in range(nPoints)]
    # 构建KD-Tree，并生成所有距离<=eps的点集合
    kd = KDTree(dataSet)
    while(vPoints.unvisitednum>0):
        p = random.choice(vPoints.unvisitedlist)
        vPoints.visit(p)
        N = kd.query_ball_point(dataSet[p], eps)
        if len(N) >= minPts:
            k += 1
            C[p] = k
            for p1 in N:
                if p1 in vPoints.unvisitedlist:
                    vPoints.visit(p1)
                    M = kd.query_ball_point(dataSet[p1], eps)
                    if len(M) >= minPts:
                        for i in M:
                            if i not in N:
                                N.append(i)
                    if C[p1] == -1:
                        C[p1] = k
        else:
            C[p1] = -1
    return C

def dbscan_lib(dataSet, eps, minPts):
    """
    @brief      利用sklearn包计算DNSCAN
    @param      dataSet  The data set
    @param      eps      The eps
    @param      minPts   The minimum points
    @return     { description_of_the_return_value }
    """
    from sklearn.cluster import DBSCAN
    C = DBSCAN(eps = eps, min_samples = minPts).fit_predict(dataSet)
    return C

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    X1, Y1 = datasets.make_circles(n_samples=2000, factor=0.6, noise=0.05,
                                   random_state=1)
    X2, Y2 = datasets.make_blobs(n_samples=500, n_features=2, centers=[[1.5,1.5]],
                                 cluster_std=[[0.1]], random_state=5)
    X = np.concatenate((X1, X2))
    plt.figure(figsize=(15, 6), dpi=80)
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], marker='.')

    # 获取集合簇
    plt.subplot(122)
    start = time.time()
    #C = dbscan_simple(X, 0.1, 10)  #31.79s
    #C = dbscan(X, 0.1, 10)    # 4.17s
    C = dbscan_lib(X, 0.1, 10)   # 0.19s
    end = time.time()
    print(end-start)
    colors = ["r", "g", "b", 'c', 'm', 'y', 'k', 'w']
    for i in range(X.shape[0]):
        plt.scatter(X[i][0], X[i][1], c = colors[C[i] % len(colors)])
    plt.show()
