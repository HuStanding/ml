# k_means.py
# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-29 09:31:43
# @Last Modified by:   huzhu
# @Last Modified time: 2019-10-31 14:44:15

import codecs
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def get_closest_dist(point, centroid):
	"""
	@brief      Gets the closest distance.
	@param      point     The point
	@param      centroid  The centroid
	@return     The closest distance.
	"""
	# 计算与已有质心最近的距离
	min_dist = inf
	for j in range(len(centroid)):
		distance = dist_eucl(point, centroid[j])
		if distance < min_dist:
			min_dist = distance
	return min_dist

def kpp_cent(data_mat, k):
	"""
	@brief      kmeans++ init centor
	@param      data_mat  The data matrix
	@param      k   num of cluster      
	@return     init centroid
	"""
	data_set = data_mat.getA()
	# 随机初始化第一个中心点
	centroid = list()
	centroid.append(data_set[random.randint(0,len(data_set))])
	d = [0 for i in range(len(data_set))]
	for _ in range(1, k):
		total = 0.0
		for i in range(len(data_set)):
			d[i] = get_closest_dist(data_set[i], centroid)
			total += d[i]
		total *= random.rand()
		# 选取下一个中心点
		for j in range(len(d)):
			total -= d[j]
			if total > 0:
				continue
			centroid.append(data_set[j])
			break
	return mat(centroid)

def kpp_Means(data_mat, k, dist = "dist_eucl", create_cent = "kpp_cent"):
	"""
	@brief      kpp means algorithm
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
	plt.title("Kpp-Means Cluster, k = 3")
	plt.show()


if __name__ == '__main__':
	#data_mat = mat(load_data("data/testSet_kmeans.txt"))
	data_mat = mat(load_data("data/testSet2_kmeans.txt"))
	centroid, cluste_assment = kpp_Means(data_mat, 3)
	plot_cluster(data_mat, cluste_assment, centroid)

	