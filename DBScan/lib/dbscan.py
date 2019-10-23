# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-22 20:37:58
# @Last Modified by:   huzhu
# @Last Modified time: 2019-10-23 19:40:25

from scipy.spatial import KDTree
import codecs
import time
import random
import math
import numpy as np
import kd_tree

class DBScan():
    """
        基于小区的DBScan聚类算法
    """
    def __init__(self, data_set, eps = 500, minPts = 10, metric = "dist"):
        """
        @brief      构造函数
        @param      self     
        @param      data_set  数据集，格式为np.array [[float1, float2],[float3,float4],...]
        @param      eps      临界距离
        @param      minPts   聚类的最小点数
        @param      metric   距离函数
        """
        self.data_set = data_set
        self.eps = eps
        self.minPts = minPts
        self.metric = metric
        #print(getattr(self, metric)([1,2],[3,4]))
    
    def predict(self):
        """
        @brief      基于kd-tree的DBScan算法
        @param      data_set  输入数据集
        @param      eps      最短距离
        @param      minPts   最小簇点数
        @return     分类标签
        """
        # 输入数据点的个数
        nPoints = len(self.data_set)
        # (1) 标记所有对象为unvisited
        vPoints = visitlist(count=nPoints)
        # (2) 初始化簇标记列表C，簇标记为 k
        k = -1
        C = [-1 for i in range(nPoints)]
        # 构建KD-Tree，并生成所有距离<=eps的点集合
        kd = KDTree(self.data_set)

        while(vPoints.unvisitednum > 0):
            # (3) 随机选择一个unvisited对象p
            p = random.choice(vPoints.unvisitedlist)
            # (4) 标t己p为visited
            vPoints.visit(p)
            # (5) 如果 p 的邻域至少有MinPts个对象
            N = kd.query_ball_point(self.data_set[p], self.eps)  # 邻域
            if len(N) >= self.minPts:
                # (6) 创建个一个新簇C，并把p添加到C
                k += 1
                C[p] = k
                # (7) 令N为p的邻域中的对象的集合
                # (8) for N中的每个点p'
                for p1 in N:
                    # (9) if p'是unvisited
                    if p1 in vPoints.unvisitedlist:
                        # (10) 标记p'为visited
                        vPoints.visit(p1)
                        # (11) if p'的$\varepsilon$-邻域至少有MinPts个点，把这些点添加到N
                        # 找出p'的邻域点，并将这些点去重新添加到N
                        M = kd.query_ball_point(self.data_set[p1], self.eps)
                        if len(M) >= self.minPts:
                            for i in M:
                                if i not in N:
                                    N.append(i)
                        # (12) if p'还不是任何簇的成员，把p'添加到c
                        if C[p1] == -1:
                            C[p1] = k
            # (13) else标记p为噪声
            else:
                C[p1] = -1

        # (14) until没有标记为unvisited的对象
        return C

    def dist(self, a, b):
        """
        @brief      { function_description }
        @param      self  The object
        @param      a     { parameter_description }
        @param      b     { parameter_description }
        @return     { description_of_the_return_value }
        """
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        

class visitlist:
    def __init__(self, count=0):
        self.unvisitedlist = [i for i in range(count)]
        self.visitedlist = list()
        self.unvisitednum = count

    def visit(self, pointId):
        self.visitedlist.append(pointId)
        self.unvisitedlist.remove(pointId)
        self.unvisitednum -= 1


if __name__ == '__main__':
    data = list()
    with codecs.open("dbscan_test.txt", encoding='utf-8') as f:
        for line in f.readlines():
            item = [float(x) for x in line.split("\t")]
            data.append(item)
    data = np.array(data)
    print(np.shape(data)[0])
    start = time.time()
    #dbscan = DBScan(data)
    #labels = dbscan.predict()
    kd_tree.fib(40)
    end = time.time()
    print(end - start)
