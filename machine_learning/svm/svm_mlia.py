# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-10-10 09:49:17
# @Last Modified by:   huzhu
# @Last Modified time: 2019-10-14 17:44:09

import codecs
import random
from numpy import *


def load_dataSet(filename):
	"""
	@brief      加载数据
	@param      filename  文件名
	@return     数据和标签
	"""
	data_mat = []
	label_mat = []
	with codecs.open(filename, "r") as f:
		for line in f.readlines():
			line_arr = line.strip().split("\t")
			data_mat.append([float(line_arr[0]), float(line_arr[1])])
			label_mat.append(float(line_arr[2]))
	return data_mat, label_mat

def select_j_rand(i ,m):
	"""
	@brief      随机选取j的值
	@param      i     alpha_i的下标
	@param      m     alpha向量的维度
	@return     下标j
	"""
	j = i
	while(j == i):
		j = int(random.uniform(0, m))
	return j

def clap_alpha(aj, H, L):
	"""
	@brief      归一化alpha_j的值
	@param      aj    归一化之前的alpha_j的值
	@param      H     最大值
	@param      L     最小值
	@return     归一化之后alpha_j的值
	"""
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj

def smo_simple(data_mat_in, class_labels, C, toler, max_iter):
	"""
	@brief      简易SMO算法
	@param      data_mat_in   输入的数据
	@param      class_labels  输入数据标签
	@param      C             惩罚参数
	@param      toler         经度
	@param      max_iter      最大迭代次数
	@return     alphas值和b值
	"""
	data_matrix = mat(data_mat_in)
	label_mat = mat(class_labels).transpose()
	b = 0    # 初始化常数项
	m, n = shape(data_matrix)  # 获取输入数据的个数和维度
	alphas = mat(zeros((m, 1)))  # 初始化alphas的值为0
	iter = 0 
	while(iter < max_iter):
		alpha_pairs_changed = 0   # 记录alpha是否被优化
		for i in range(m):
			# 计算预测值
			fXi = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
			# 计算误差值
			Ei = fXi - float(label_mat[i])
			# 如果alpha可以更改进入优化过程
			if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or \
				((label_mat[i] * Ei > toler) and (alphas[i] > 0)):
				# 随机选择j的值
				j = select_j_rand(i, m)
				fXj = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
				# 计算误差值
				Ej = fXj - float(label_mat[j])
				alpha_Iold = alphas[i].copy()
				alpha_Jold = alphas[j].copy()
				# 确定最优值的取值上下界
				if (label_mat[i] != label_mat[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0, alphas[j] + alphas[i] - C)
					H = min(C, C + alphas[j] + alphas[i])
				if L == H:
					print("L == H")
					continue

				# 计算η的值
				eta = 2 * data_matrix[i, :] * data_matrix[j, :].T - \
					  data_matrix[i, :] * data_matrix[i, :].T -data_matrix[j, :] * data_matrix[j, :].T
				if eta >= 0:
					print("eta >= 0")
					continue

				# 计算新的alpha的值
				alphas[j] -= label_mat[j] * (Ei - Ej) / eta
				alphas[j] = clap_alpha(alphas[j], H, L)
				if abs(alphas[j] - alpha_Jold) < 0.0001:
					print("j not moving enough")
					continue
				alphas[i] = alpha_Iold + label_mat[j] * label_mat[i] * (alpha_Jold - alphas[j])

				# 更新b的值
				b1 = b - Ei - label_mat[i] *(alphas[i] - alpha_Iold) * data_matrix[i, :] * data_matrix[i, :].T \
					 - label_mat[j] * (alphas[j] - alpha_Jold) * data_matrix[i,: ] * data_matrix[j, :].T
				b2 = b - Ej - label_mat[i] *(alphas[i] - alpha_Iold) * data_matrix[i, :] * data_matrix[j, :].T \
					 - label_mat[j] * (alphas[j] - alpha_Jold) * data_matrix[j,: ] * data_matrix[j, :].T
				if (alphas[i] > 0) and (alphas[i] < C):
					b = b1
				elif (alphas[j] > 0) and (alphas[j] < C):
					b = b2
				else:
					b = (b1 + b2) / 2

				# 增加迭代次数
				alpha_pairs_changed += 1 
				print("iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
		if alpha_pairs_changed == 0:
			iter += 1 
		else:
			print("iteration number: %d" % iter)
	return b, alphas

class optStruct():
	"""
	@brief      存储一些关键数据
	"""
	def __init__(self, data_matin, class_labels, C, toler):
		self.X = data_matin
		self.label_mat = class_labels
		self.C = C
		self.tol = toler
		self.m = shape(data_matin)[0]
		self.alphas = mat(zeros((self.m , 1)))
		self.b = 0
		# 误差缓存,第一列是eCache是否有效的标志位，第二列是实际的E值
		self.eCache = mat(zeros((self.m , 2)))

def calc_Ek(os, k):
	"""
	@brief      计算Ek的值
	@param      os    输入的数据对象
	@param      k	  下标
	@return     Ek的值
	"""
	fXk = float(multiply(os.alphas, os.label_mat).T * (os.X * os.X[k,:].T)) + os.b
	Ek = fXk - float(os.label_mat[k])
	return Ek

def select_j(i, os, Ei):
	"""
	@brief      选取j的下标值
	@param      i     下标
	@param      os    操作对象
	@param      Ei    Ei值
	@return     j, Ej
	"""
	max_K = -1
	max_deltaE = 0
	Ej = 0
	os.eCache[i] = [1, Ei]
	valid_eCache_list = nonzero(os.eCache[:, 0].A)[0]
	if len(valid_eCache_list) > 1:
		for k in valid_eCache_list:
			if k == i:
				continue
			Ek = calc_Ek(os, k)
			deltaE = abs(Ei - Ek)
			# 选取具有最大步长的j值
			if (deltaE > max_deltaE):
				max_K = k
				max_deltaE = deltaE
				Ej = Ek
			return max_K, Ej
	else:
		j = select_j_rand(i, os.m)
		Ej = calc_Ek(os, j)
	return j, Ej

def update_Ek(os, k):
	"""
	@brief      计算误差值并存入缓存中
	@param      os    
	@param      k     
	@return     
	"""
	Ek = calc_Ek(os, k)
	os.eCache[k] = [1, Ek]

def innerL(i, os):
	"""
	@brief      Platt SMO算法的优化例程
	@param      i     { parameter_description }
	@param      os    The operating system
	@return     { description_of_the_return_value }
	"""
	Ei = calc_Ek(os ,i)
	if ((os.label_mat[i] * Ei < -os.tol) and (os.alphas[i] < os.C)) or \
	   ((os.label_mat[i] * Ei > os.tol) and (os.alphas[i] > 0)):
		# 随机选择j的值
		j, Ej = select_j(i, os, Ei)

		alpha_Iold = os.alphas[i].copy()
		alpha_Jold = os.alphas[j].copy()

		if (os.label_mat[i] != os.label_mat[j]):
			L = max(0, os.alphas[j] - os.alphas[i])
			H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
		else:
			L = max(0, os.alphas[j] + os.alphas[i] - os.C)
			H = min(os.C, os.C + os.alphas[j] + os.alphas[i])
		if L == H:
			print("L == H")
			return 0

		# 计算η的值
		eta = 2 * os.X[i, :] * os.X[j, :].T - os.X[i, :] * os.X[i, :].T -os.X[j, :] * os.X[j, :].T
		if eta >= 0:
			print("eta >= 0")
			return 0

		# 计算新的alpha的值
		os.alphas[j] -= os.label_mat[j] * (Ei - Ej) / eta
		os.alphas[j] = clap_alpha(os.alphas[j], H, L)
		update_Ek(os, j)

		if abs(os.alphas[j] - alpha_Jold) < 0.0001:
			print("j not moving enough")
			return 0
		os.alphas[i] = alpha_Iold + os.label_mat[j] * os.label_mat[i] * (alpha_Jold - os.alphas[j])
		update_Ek(os, i)

		# 更新b的值
		b1 = os.b - Ei - os.label_mat[i] *(os.alphas[i] - alpha_Iold) * os.X[i, :] * os.X[i, :].T \
			 - os.label_mat[j] * (os.alphas[j] - alpha_Jold) * os.X[i,: ] * os.X[j, :].T
		b2 = os.b - Ej - os.label_mat[i] *(os.alphas[i] - alpha_Iold) * os.X[i, :] * os.X[j, :].T \
			 - os.label_mat[j] * (os.alphas[j] - alpha_Jold) * os.X[j,: ] * os.X[j, :].T
		if (os.alphas[i] > 0) and (os.alphas[i] < os.C):
			os.b = b1
		elif (os.alphas[j] > 0) and (os.alphas[j] < os.C):
			os.b = b2
		else:
			os.b = (b1 + b2) / 2
		return 1
	else:
		return 0


def smoP(data_matin, class_labels, C, toler, max_iter, kTup = ("1in", 0)):
	"""
	@brief      完整的SMO算法
	@param      data_matin    The data matin
	@param      class_labels  The class labels
	@param      C             { parameter_description }
	@param      toler         The toler
	@param      max_iter      The maximum iterator
	@param      kTup          The k tup
	@return     b, alphas
	"""
	os = optStruct(mat(data_matin), mat(class_labels).transpose(), C, toler)
	iter = 0 
	entire_set = True
	alpha_pairs_changed = 0
	while (iter < max_iter) and ((alpha_pairs_changed > 0) or (entire_set)):
		alpha_pairs_changed = 0
		if entire_set:
			for i in range(os.m):
				alpha_pairs_changed += innerL(i, os)
				print("fullset, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed)) 
			iter += 1 
		else:
			non_boundIs = nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
			for i in non_boundIs:
				alpha_pairs_changed += innerL(i, os)
				print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed)) 
			iter += 1 
		if entire_set:
			entire_set = False
		elif alpha_pairs_changed == 0:
			entire_set = True
		print("iteration number: %d" % iter)
	return os.b, os.alphas

def calc_ws(alphas, data_arr, class_labels):
	"""
	@brief      计算w的值
	@param      alphas        The alphas
	@param      data_arr      The data arr
	@param      class_labels  The class labels
	@return     The ws.
	"""
	X = mat(data_arr)
	label_mat = mat(class_labels).transpose()
	m, n = shape(X)
	w = zeros((n, 1))
	for i in range(m):
		w += multiply(alphas[i] * label_mat[i, :], X[i, :].T)
	return w

if __name__ == '__main__':
	data_arr, label_arr = load_dataSet("testSet.txt")

	b, alphas = smoP(data_arr, label_arr, 0.6, 0.001, 40)
	print(b,alphas[alphas>0])
	print(shape(alphas[alphas > 0]))
	for i in range(100):
		if alphas[i] > 0.0:
			print(data_arr[i], label_arr[i])

	w = calc_ws(alphas, data_arr, label_arr)
	print(w)
	# 检查数据的正确性
	data_mat = mat(data_arr)
	lable = data_mat[49] * mat(w) + b
	print (lable)
	a = mat([1,2,3,4])
	print(a)


