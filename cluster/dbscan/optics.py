# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-11-08 09:15:57
# @Last Modified by:   huzhu
# @Last Modified time: 2019-11-12 09:59:40
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

def dist(a, b):
    """
    @brief      计算a,b两个元组的欧几里得距离
    @param      a
    @param      b
    @return     距离
    """
    return math.sqrt(np.power(a-b, 2).sum())


def load_data():
	"""
	@brief      加载一个密度不均的数据
	@return     { description_of_the_return_value }
	"""
	X1, Y1 = datasets.make_blobs(n_samples=400, n_features=2, centers=[
	                             [10, 8]], cluster_std=[[1]], random_state=1)
	X2, Y2 = datasets.make_blobs(n_samples=200, n_features=2, centers=[
	                             [2, 2]], cluster_std=[[2]], random_state=5)
	X3, Y3 = datasets.make_blobs(n_samples=200, n_features=2, centers=[
	                             [-5, 20]], cluster_std=[[3]], random_state=4)
	X = np.concatenate((X1, X2))
	X = np.concatenate((X, X3))
	return X

def dbscan_lib(dataSet, eps, minPts):
    """
    @brief      利用sklearn包计算DNSCAN
    @param      dataSet  The data set
    @param      eps      The eps
    @param      minPts   The minimum points
    @return     { description_of_the_return_value }
    """
    from sklearn.cluster import DBSCAN
    label = DBSCAN(eps = eps, min_samples = minPts).fit_predict(dataSet)
    return label

class Optics(object):
    """Optics算法"""
    def __init__(self, dataset):
        self.dataset = dataset 
        self.n = len(dataset)
        self.unvisited = [i for i in range(self.n)]
        self.visited = list()
        self.order_list = list()

    def visit(self, index):
        self.visited.append(index)
        self.unvisited.remove(index)
        self.order_list.append(index)

    def cal_core_dist(self, point, point_neighbors, min_pts):
        # 按照离points点的距离排序
        sorted_dist = sorted([dist(self.dataset[point], self.dataset[item]) for item in point_neighbors])
        return sorted_dist[min_pts - 1]

    def optics(self, eps = 0.1, min_pts = 5):
        self.eps = eps
        self.reach_dist = [inf for i in range(self.n)]      # 可达距离
        self.core_dist = [inf for i in range(self.n)]     # 核心距离
        kd = KDTree(self.dataset)
        while(self.unvisited):
            # 随机选取一个点
            i = random.choice(self.unvisited)
            self.visit(i)
            # 获取i的邻域
            neighbors_i = kd.query_ball_point(self.dataset[i], eps)
            # 如果i是核心点
            if len(neighbors_i) >= min_pts:
                # 计算核心距离
                self.core_dist[i] = self.cal_core_dist(i, neighbors_i, min_pts)
                seed_list = list()
                self.insert_list(i, neighbors_i,seed_list)
                while(seed_list):
                    seed_list.sort(key=lambda x:self.reach_dist[x])
                    j = seed_list.pop(0)
                    self.visit(j)
                    neighbors_j = kd.query_ball_point(self.dataset[j], eps)
                    if len(neighbors_j) >= min_pts:
                        self.core_dist[j] = self.cal_core_dist(j, neighbors_j, min_pts)
                        self.insert_list(j, neighbors_j,seed_list)
        return self.order_list, self.reach_dist

    def insert_list(self, point, point_neighbors, seed_list):
        for i in point_neighbors:
            if i in self.unvisited:
                rd = max(self.core_dist[point], dist(self.dataset[i], self.dataset[point]))
                if self.reach_dist[i] == inf:
                    self.reach_dist[i] = rd
                    seed_list.append(i)
                elif rd < self.reach_dist[i]:
                    self.reach_dist[i] = rd
                    

    def extract(self, cluster_threshold):
        clsuter_id = -1
        label = [-1 for i in range(self.n)]
        k = 0
        for i in range(self.n):
            j = self.order_list[i]
            if self.reach_dist[j] > cluster_threshold:
                if self.core_dist[j] <= cluster_threshold:
                    clsuter_id = k
                    k += 1 
                    label[j] = clsuter_id
                else:
                    label[j] = -1
            else:
                label[j] = clsuter_id
        return label


def plot_test():
    X = load_data()
    plt.figure(figsize=(15, 6), dpi=80)

    plt.subplot(121)
    test = Optics(X)
    order_list, reach_dist = test.optics(eps = 10, min_pts = 5)
    x = [reach_dist[i] for i in order_list]
    plt.bar(range(len(order_list)),x)
    plt.title("optics, core_distance of every order_list element", fontsize=15)

    plt.subplot(122)
    label = np.array(test.extract(2.2))
    k = len(set(label))
    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(label == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markersize=5)
    plt.title("optics, eps = 10, min_pts = 5, eps' = 2.2", fontsize=15)
    plt.show()

def plot_dbscan():
    X = load_data()
    plt.figure(figsize=(15, 6), dpi=80)
    plt.subplot(121)
    plt.plot(X[:, 0], X[:, 1], 'o',markersize=5)
    plt.title("source data", fontsize=15)
    
    plt.subplot(122)
    label = dbscan_lib(X, 1.5, 5)
    k = len(set(label))
    colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
    for i, col in zip(range(k), colors):
        per_data_set = X[nonzero(label == i - 1)[0]]
        plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),markersize=5)
    plt.title("dbscan, eps = 1.5, min_pts = 5", fontsize=15)
    plt.show()

if __name__ == '__main__':
    plot_test()
    #plot_dbscan()

    
