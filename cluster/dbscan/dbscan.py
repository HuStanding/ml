# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-14 20:51:30
# @Last Modified by:   huzhu
# @Last Modified time: 2019-11-13 11:37:26

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from scipy.spatial import KDTree
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.animation as animation
import copy

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
            C[p] = -1
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

def plot_test():
    # X1, Y1 = datasets.make_circles(n_samples=2000, factor=0.6, noise=0.05,
    #                                random_state=1)
    # X2, Y2 = datasets.make_blobs(n_samples=500, n_features=2, centers=[[1.5,1.5]],
    #                              cluster_std=[[0.1]], random_state=5)

    # X = np.concatenate((X1, X2))
    X,c = make_moons(n_samples=1000,noise=0.1) 
    plt.figure(figsize=(15, 6), dpi=80)
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], marker='.')
    plt.title("source data", fontsize=15)


    # 获取集合簇
    plt.subplot(122)
    C = dbscan_lib(X, 0.1, 5)
    k = len(set(C))
    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(C == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markersize=5)
    plt.title("eps = 0.1, min_pts = 5, cluster = {}".format(str(k)), fontsize=15)
    plt.show()

def plot_diff_eps():
    #X = np.concatenate((X1, X2))
    X,c = make_moons(n_samples=1000,noise=0.1)  
    plt.figure(figsize=(15, 12), dpi=80)
    plt.subplot(221)
    C = dbscan_lib(X, 0.05, 5)
    k = len(set(C))
    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(C == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markersize=5)
    plt.title("eps = 0.05, min_pts = 5, cluster = {}".format(str(k)), fontsize=15)

    plt.subplot(222)
    C = dbscan_lib(X, 0.08, 5)
    k = len(set(C))
    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(C == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markersize=5)
    plt.title("eps = 0.1, min_pts = 5, cluster = {}".format(str(k)), fontsize=15)

    plt.subplot(223)
    C = dbscan_lib(X, 0.1, 5)
    k = len(set(C))
    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(C == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markersize=5)
    plt.title("eps = 0.15, min_pts = 5, cluster = {}".format(str(k)), fontsize=15)

    plt.subplot(224)
    C = dbscan_lib(X, 0.15, 5)
    k = len(set(C))
    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(C == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col), markersize=5)
    plt.title("eps = 0.2, min_pts = 5, cluster = {}".format(str(k)), fontsize=15)
    plt.show()

def plot_diff_minpts():
    #X = np.concatenate((X1, X2))
    X,c = make_moons(n_samples=1000,noise=0.1)  
    plt.figure(figsize=(15, 12), dpi=80)
    plt.subplot(221)
    C = dbscan_lib(X, 0.1, 5)
    k = len(set(C))
    print(k)
    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(C == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markersize=5)
    plt.title("eps = 0.1, min_pts = 5, cluster = {}".format(str(k)), fontsize=15)


    plt.subplot(222)
    C = dbscan_lib(X, 0.1, 10)
    k = len(set(C))
    print(k)

    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(C == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markersize=5)
    plt.title("eps = 0.1, min_pts = 10, cluster = {}".format(str(k)), fontsize=15)

    plt.subplot(223)
    C = dbscan_lib(X, 0.1, 15)
    k = len(set(C))
    print(k)

    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(C == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markersize=5)
    plt.title("eps = 0.1, min_pts = 15, cluster = {}".format(str(k)), fontsize=15)

    plt.subplot(224)
    C = dbscan_lib(X, 0.1, 20)
    k = len(set(C))
    print(k)

    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(C == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col), markersize=5)
    plt.title("eps = 0.1, min_pts = 20, cluster = {}".format(str(k)), fontsize=15)
    plt.show()

def plot_fig():
    """
    @brief      绘制并保存gif图
    @param      k         { parameter_description }
    @return     { description_of_the_return_value }
    """
    data_mat,c = make_moons(n_samples=1000,noise=0.1) 
    C_list = list()
    def sub_dbscan(dataSet, eps, minPts):
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
                        new_C = C.copy()
                        C_list.append(new_C)
                        if len(M) >= minPts:
                            for i in M:
                                if i not in N:
                                    N.append(i)
                        if C[p1] == -1:
                            C[p1] = k
                            
            else:
                C[p] = -1
        return C_list
    C_list = sub_dbscan(data_mat,0.1,3)
    C_list = [C_list[i] for i in range(0, len(C_list), 20)]
    # 绘制动图
    fig, ax = plt.subplots()
    plt.scatter(data_mat[:, 0], data_mat[:, 1], c = 'k')
    plt.title("DBSCAN Cluster Process", fontsize=15)
    def update(i):
        try:
            ax.lines.pop()
        except Exception:
            pass
        C = array(C_list[i])
        k = len(set(C))
        colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
        for i, col in zip(range(k), colors):
            per_data_set = data_mat[nonzero(C == i)[0]]
            line, =plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=10)
        return line,

    anim = animation.FuncAnimation(fig, update, frames=len(C_list),interval=100, repeat=False)
    #plt.show() 
    anim.save('../pic/dbscan_process.gif',writer='pillow')

if __name__ == '__main__':
    #plot_diff_minpts()
    #plot_diff_eps()
    plot_test()
    #plot_fig()
    
    
