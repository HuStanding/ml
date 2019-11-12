# k_means.py
# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-29 09:31:43
# @Last Modified by:   huzhu
# @Last Modified time: 2019-11-11 21:22:57

import codecs
from numpy import *
import matplotlib.pyplot as plt

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

def dist_eucl(vecA, vecB):
	"""
	@brief      the similarity function
	@param      vecA  The vector a
	@param      vecB  The vector b
	@return     the euclidean distance
	"""
	return sqrt(sum(power(vecA - vecB, 2)))

def k_Means(data_mat, k, dist = "dist_eucl", create_cent = "rand_cent"):
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
	cluster_assment = mat(zeros((m, 2)))  # 类别，距离
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
			if cluster_assment[i, 0] != min_index:
				cluster_changed = True
			cluster_assment[i, :] = min_index, min_dist**2
		# 计算簇中所有点的均值并重新将均值作为质心
		for j in range(k):
			per_data_set = data_mat[nonzero(cluster_assment[:,0].A == j)[0]]
			centroid[j, :] = mean(per_data_set, axis = 0)
	return centroid, cluster_assment

def bi_kMeans(data_mat, k, dist = "dist_eucl"):
	"""
	@brief      kMeans algorithm
	@param      data_mat     The data matrix
	@param      k            num of cluster
	@param      dist         The distance funtion
	@return     the cluster
	"""
	m = shape(data_mat)[0]

	# 初始化点的簇
	cluster_assment = mat(zeros((m, 2)))  # 类别，距离

	# 初始化聚类初始点
	centroid0 = mean(data_mat, axis = 0).tolist()[0]
	cent_list = [centroid0]
	print(cent_list)

	# 初始化SSE
	for j in range(m):
		cluster_assment[j, 1] = eval(dist)(mat(centroid0), data_mat[j, :]) ** 2 

	while (len(cent_list) < k):
		lowest_sse = inf 
		for i in range(len(cent_list)):
			# 尝试在每一类簇中进行k=2的kmeans划分
			ptsin_cur_cluster = data_mat[nonzero(cluster_assment[:, 0].A == i)[0],:]
			centroid_mat, split_cluster_ass = k_Means(ptsin_cur_cluster,k = 2)
			# 计算分类之后的SSE值
			sse_split = sum(split_cluster_ass[:, 1])
			sse_nonsplit = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])
			print("sse_split, sse_nonsplit", sse_split, sse_nonsplit)
			# 记录最好的划分位置
			if sse_split + sse_nonsplit < lowest_sse:
				best_cent_tosplit = i
				best_new_cents = centroid_mat
				best_cluster_ass = split_cluster_ass.copy()
				lowest_sse = sse_split + sse_nonsplit
		print( 'the bestCentToSplit is: ', best_cent_tosplit)
		print ('the len of bestClustAss is: ', len(best_cluster_ass))
		# 更新簇的分配结果
		best_cluster_ass[nonzero(best_cluster_ass[:, 0].A == 1)[0], 0] = len(cent_list)
		best_cluster_ass[nonzero(best_cluster_ass[:, 0].A == 0)[0], 0] = best_cent_tosplit
		cent_list[best_cent_tosplit] = best_new_cents[0, :].tolist()[0]
		cent_list.append(best_new_cents[1, :].tolist()[0])
		cluster_assment[nonzero(cluster_assment[:, 0].A == best_cent_tosplit)[0],:] = best_cluster_ass
	return mat(cent_list), cluster_assment

def plot_cluster(data_mat, cluster_assment, centroid):
	"""
	@brief      plot cluster and centroid
	@param      data_mat        The data matrix
	@param      cluster_assment  The cluste assment
	@param      centroid        The centroid
	@return     
	"""
	plt.figure(figsize=(15, 6), dpi=80)
	plt.subplot(121)
	plt.plot(data_mat[:, 0], data_mat[:, 1], 'o')
	plt.title("source data", fontsize=15)
	plt.subplot(122)
	k = shape(centroid)[0]
	colors = [plt.cm.Spectral(each) for each in linspace(0, 1, k)]
	for i, col in zip(range(k), colors):
	    per_data_set = data_mat[nonzero(cluster_assment[:,0].A == i)[0]]
	    plt.plot(per_data_set[:, 0], per_data_set[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=10)
	for i in range(k):
		plt.plot(centroid[:,0], centroid[:,1], '+', color = 'k', markersize=18)
	plt.title("bi_KMeans Cluster, k = 3", fontsize=15)
	plt.show()

if __name__ == '__main__':
	# data_mat = mat(load_data("data/testSet_kmeans.txt"))
	data_mat = mat(load_data("data/testSet2_kmeans.txt"))
	centroid, cluster_assment = bi_kMeans(data_mat, 3)
	sse = sum(cluster_assment[:,1])
	print("sse is ", sse)
	plot_cluster(data_mat, cluster_assment, centroid)
	# plot_noncov()
	# test_diff_k()

	
