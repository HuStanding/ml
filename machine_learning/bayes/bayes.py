# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-09-19 20:43:50
# @Last Modified by:   huzhu
# @Last Modified time: 2019-09-24 21:11:40
# @description p(ci|w) = p(w|ci)p(ci)/p(w)

from numpy import *

def load_dataSet():
	"""
	@brief      加载数据集
	@return     训练集，分类标签
	"""
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
	return postingList,classVec

def create_vocablist(dataSet):
	"""
	@brief      Creates a vocablist.
	@param      dataSet  The data set
	@return     { description_of_the_return_value }
	"""
	vocabSet = set([])
	for doc in dataSet:
		vocabSet = vocabSet | set(doc)  #求解并集
	return list(vocabSet)


def set_of_words2vec(vocablist, inputset):
	"""
	@brief      Sets the of words 2 vector.
	@param      vocablist  The vocablist
	@param      inputset   The inputset
	@return     { description_of_the_return_value }
	"""
	res = [0] * len(vocablist)
	for word in inputset:
		if word in vocablist:
			res[vocablist.index(word)] = 1
		else:
			print("the word: %s is not in my vocabulary!") % word 
	return res 

def train(train_matrix, train_category):
	"""
	@brief      训练函数
	@param      train_matrix    The train matrix
	@param      train_category  The train category
	@return     { description_of_the_return_value }
	"""
	num_train = len(train_matrix)
	num_words = len(train_matrix[0])
	p0 = ones(num_words)
	p1 = ones(num_words)
	# 计算辱骂性段落占比，即p(c1)
	pAbusive = sum(train_category) / float(num_train)
	p0_num = p1_num = 2.0
	for i in range(num_train):
		if train_category[i] == 0:
			p0 += train_matrix[i]
			p0_num += sum(train_matrix[i])
		else:
			p1 += train_matrix[i]
			p1_num += sum(train_matrix[i])
	p0 = log(p0 / p0_num)
	p1 = log(p1 / p1_num)
	return p0, p1, pAbusive

def classify(vec2classify, p0, p1, p_class1):
	"""
	@brief      { function_description }
	@param      vec2classify  The vector 2 classify
	@param      p0            The p 0
	@param      p1            The p 1
	@param      p_class1      The class 1
	@return     { description_of_the_return_value }
	"""
	p0 = sum(vec2classify * p0) + log(1 - p_class1)
	p1 = sum(vec2classify * p1) + log(p_class1)
	if p0 > p1:
		return 0
	else:
		return 1

def test():
	list_posts, list_class = load_dataSet()
	my_vocablist = create_vocablist(list_posts)
	train_matrix = []
	for doc in list_posts:
		train_matrix.append(set_of_words2vec(my_vocablist, doc))
	p0,p1,pAb = train(train_matrix, list_class)
	test_entry = ["stupid", "garbage", "dalmation"]
	this_doc = set_of_words2vec(my_vocablist,test_entry)
	print("test_entry is class ", classify(this_doc, p0, p1, pAb))


if __name__ == '__main__':
	test()


