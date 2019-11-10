# k_means.py
# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-29 09:31:43
# @Last Modified by:   huzhu
# @Last Modified time: 2019-11-05 10:02:37

import codecs
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons

def load_data(path):
	"""
	@brief      Loads a data.
	@param      path  The path
	@return     data set
	"""
	data_set = list()
	with codecs.open(path) as f:
		for line in f.readlines():
			data = line.strip().split("\t")
			flt_data = list(map(float, data))
			data_set.append(flt_data)
	return data_set

def dist_eucl(vecA, vecB):
	"""
	@brief      the similarity function
	@param      vecA  The vector a
	@param      vecB  The vector b
	@return     the euclidean distance
	"""
	return sqrt(sum(power(vecA - vecB, 2)))

def kMeans(data_mat, k, dist = "dist_eucl", create_cent = "rand_cent"):
	"""
	@brief      kMeans algorithm
	@param      data_mat     The data matrix
	@param      k            num of cluster
	@param      dist         The distance funtion
	@param      create_cent  The create centroid function
	@return     the cluster
	"""
	m = shape(data_mat)[0]
	# 初始化点的簇
	cluste_assment = mat(zeros((m, 2)))  # 类别，距离
	# 随机初始化聚类初始点
	centroid = eval(create_cent)(data_mat, k)
	cluster_changed = True
	# 遍历每个点
	while cluster_changed:
		cluster_changed = False
		for i in range(m):
			min_index = -1
			min_dist = inf
			for j in range(k):
				distance = eval(dist)(data_mat[i, :], centroid[j, :])
				if distance < min_dist:
					min_dist = distance
					min_index = j
			if cluste_assment[i, 0] != min_index:
				cluster_changed = True
				cluste_assment[i, :] = min_index, min_dist**2
		# 计算簇中所有点的均值并重新将均值作为质心
		for j in range(k):
			per_data_set = data_mat[nonzero(cluste_assment[:,0].A == j)[0]]
			centroid[j, :] = mean(per_data_set, axis = 0)
	return centroid, cluste_assment

def plot_cluster(data_mat, cluste_assment, centroid):
	"""
	@brief      plot cluster and centroid
	@param      data_mat        The data matrix
	@param      cluste_assment  The cluste assment
	@param      centroid        The centroid
	@return     
	"""
	plt.figure(figsize=(15, 6), dpi=80)
	plt.subplot(121)
	plt.plot(data_mat[:, 0], data_mat[:, 1], 'o')
	plt.title("source data")
	plt.subplot(122)
	k = shape(centroid)[0]
	colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
	for i, col in zip(range(k), colors):
	    per_data_set = data_mat[nonzero(cluste_assment[:,0].A == i)[0]]
	    plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=10)
	for i in range(k):
		plt.plot(centroid[:,0], centroid[:,1], '+', color = 'k', markersize=18)
	plt.title("K-Means Cluster, k = 3")
	plt.show()

def plot_noncov():
	"""
	@brief      绘制非凸优化函数图像
	@return     { description_of_the_return_value }
	"""
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x1 = linspace(-2,2,100)
	x2 = linspace(-2,2,100)
	mu1 = array([1,1])
	mu2 = array([-1,-1])
	Z = zeros((len(x1), len(x2)))
	for i in range(len(x1)):
		for j in range(len(x2)):
			itemx = x1[i]
			itemy = x2[j]
			z1 = dist_eucl(mu1, [itemx, itemy])
			z2 = dist_eucl(mu2, [itemx, itemy])
			Z[i,j] = min(z1,z2)
	X1, X2 = meshgrid(x1, x2)
	ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='rainbow')
	plt.show()

def test_diff_k():
	plt.figure(figsize=(15, 4), dpi=80)
	data_mat = mat(load_data("data/testSet2_kmeans.txt"))
	centroid, cluste_assment = kMeans(data_mat, 2)
	plt.subplot(131)
	k = shape(centroid)[0]
	colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
	for i, col in zip(range(k), colors):
	    per_data_set = data_mat[nonzero(cluste_assment[:,0].A == i)[0]]
	    plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=10)
	for i in range(k):
		plt.plot(centroid[:,0], centroid[:,1], '+', color = 'k', markersize=18)
	plt.title("K-Means Cluster, k = 2")
	
	centroid, cluste_assment = kMeans(data_mat, 3)
	plt.subplot(132)
	k = shape(centroid)[0]
	colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
	for i, col in zip(range(k), colors):
	    per_data_set = data_mat[nonzero(cluste_assment[:,0].A == i)[0]]
	    plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=10)
	for i in range(k):
		plt.plot(centroid[:,0], centroid[:,1], '+', color = 'k', markersize=18)
	plt.title("K-Means Cluster, k = 3")

	centroid, cluste_assment = kMeans(data_mat, 4)
	plt.subplot(133)
	k = shape(centroid)[0]
	colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
	for i, col in zip(range(k), colors):
	    per_data_set = data_mat[nonzero(cluste_assment[:,0].A == i)[0]]
	    plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=10)
	for i in range(k):
		plt.plot(centroid[:,0], centroid[:,1], '+', color = 'k', markersize=18)
	plt.title("K-Means Cluster, k = 4")
	plt.show()

def rand_cent(data_mat, k):
	"""
	@brief      select random centroid
	@param      data_mat  The data matrix
	@param      k
	@return     centroids
	"""
	n = shape(data_mat)[1]
	centroids = mat(zeros((k, n)))
	for j in range(n):
		minJ = min(data_mat[:,j]) 
		rangeJ = float(max(data_mat[:,j]) - minJ)
		centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
	return centroids

if __name__ == '__main__':
	#data_mat = mat(load_data("data/testSet_kmeans.txt"))
	#data_mat = mat(load_data("data/testSet2_kmeans.txt"))
	data_mat,c = make_moons(n_samples=1000,noise=0.1)  
	centroid, cluste_assment = kMeans(data_mat, 2)
	plot_cluster(data_mat, cluste_assment, centroid)
	#plot_noncov()
	#test_diff_k()

	