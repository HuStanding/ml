# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-09-16 13:36:05
# @Last Modified by:   huzhu
# @Last Modified time: 2019-09-16 21:56:35
from math import log

def cal_shannonEnt(dataSet):
	"""
	@brief      { 计算信息熵 }
	@param      dataSet  The data set
	@return     { description_of_the_return_value }
	"""
	num_entries = len(dataSet)
	label_count = {}
	for feat_vec in dataSet:
		current_label = feat_vec[-1]
		if current_label not in label_count.keys():
			label_count[current_label] = 0
		label_count[current_label] += 1
	shannonEnt = 0.0
	for key in label_count:
		prob = float(label_count[key]) / num_entries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

def create_dataSet():
	"""
	@brief      Creates a data set.
	@return     { description_of_the_return_value }
	"""
	dataSet = [[1, 1, "yes"],
			   [1, 1, "yes"],
			   [1, 0, "no"],
			   [0, 1, "no"],
			   [0, 1, "no"]]
	labels = ["no surfacing", "flippers"]
	return dataSet, labels

def split_dataSet(dataSet, axis, value):
	"""
	@brief      划分数据集
	@param      dataSet  The data set
	@param      axis     The axis
	@param      value    The value
	@return     { description_of_the_return_value }
	"""
	res = []
	for feat_vec in dataSet:
		if feat_vec[axis] == value:
			reduced_feat_vec = feat_vec[:axis]
			reduced_feat_vec.extend(feat_vec[axis + 1 :])
			res.append(reduced_feat_vec)
	return res

def choose_best_feature_to_split(dataSet):
	"""
	@brief      { 选取最佳特征的索引 }
	@param      dataSet  The data set
	@return     { description_of_the_return_value }
	"""
	# 特征个数
	num_features = len(dataSet[0]) - 1  
	# 计算数据集的信息熵
	base_entropy = cal_shannonEnt(dataSet)
	# 定义信息增益、最优特征
	best_info_gain = 0.0
	best_feature = -1

	for i in range(num_features):
		feat_list = [example[i] for example in dataSet]
		unique_values = set(feat_list)
		new_entropy = 0.0
		for value in unique_values:
			sub_dataSet = split_dataSet(dataSet, i, value)
			prob = float(len(sub_dataSet)) / len(dataSet)
			new_entropy += prob * cal_shannonEnt(sub_dataSet)
		# 计算信息增益
		info_gain = base_entropy - new_entropy
		if info_gain > best_info_gain:
			best_info_gain = info_gain
			best_feature = i
	return best_feature

def major_cnt(class_list):
	"""
	@brief      { 获取出现次数最多的类别 }
	@param      class_list  The class list
	@return     { description_of_the_return_value }
	"""
	class_count = dict()
	for vote in class_list:
		if vote not in class_count.keys():
			class_count[vote] = 0
		class_count[vote] += 1
	sorted_class_count = sorted(class_count.items(), key=lambda d: d[1], reverse = True)
	return sorted_class_count[0][0]

def create_tree(dataSet, labels):
	"""
	@brief      创建决策树
	@param      dataSet  The data set
	@param      labels   The labels
	@return     { description_of_the_return_value }
	"""
	class_list = [example[-1] for example in dataSet]
	# 只存在一个特征，返回该特征
	if class_list.count(class_list[0]) == len(class_list):
		return class_list[0]
	# 遍历完所有特征时返回出现次数最多的类别
	if len(dataSet[0]) == 1:
		return major_cnt(class_list)
	best_feature = choose_best_feature_to_split(dataSet)
	best_label = labels[best_feature]
	my_tree = {best_label:{}}
	del(labels[best_feature])
	feat_values = [example[best_feature] for example in dataSet]
	unique_values = set(feat_values)
	for value in unique_values:
		sub_labels = labels[:]
		my_tree[best_label][value] = create_tree(split_dataSet(dataSet, best_feature, value), sub_labels)
	return my_tree

if __name__ == '__main__':
	my_data, labels = create_dataSet()
	print(cal_shannonEnt(my_data))
	print(create_tree(my_data, labels))

	# test = [1,1,1,1,2,2,4,5,9,9,9,9,9]
	# print(major_cnt(test))