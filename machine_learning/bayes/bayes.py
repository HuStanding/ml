# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-09-19 20:43:50
# @Last Modified by:   huzhu
# @Last Modified time: 2019-09-28 17:58:09
# @description p(ci|w) = p(w|ci)p(ci)/p(w)

import re
import codecs
from numpy import *
import feedparser

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


def bag_of_words2vec(vocablist, inputset):
	"""
	@brief      Sets the of words 2 vector.
	@param      vocablist  The vocablist
	@param      inputset   The inputset
	@return     { description_of_the_return_value }
	"""
	res = [0] * len(vocablist)
	for word in inputset:
		if word in vocablist:
			res[vocablist.index(word)] += 1
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
		train_matrix.append(bag_of_words2vec(my_vocablist, doc))
	p0,p1,pAb = train(train_matrix, list_class)
	test_entry = ["stupid", "garbage", "dalmation"]
	this_doc = bag_of_words2vec(my_vocablist,test_entry)
	print("test_entry is class ", classify(this_doc, p0, p1, pAb))

def text_parse(big_string):
	"""
	@brief      长字符串解析过程
	@param      big_string  The big string
	@return     处理后的字符串列表
	"""
	list_of_tokens = re.split(r"\W*", big_string)
	return [token.lower() for token in list_of_tokens if len(token) > 2]

def spam_test():
	"""
	@brief      利用贝叶斯分类器自动化处理
	@return     无
	"""
	doc_list = []
	class_list = []
	full_text = []
	for i in range(1,26):
		# 循环读取文件夹下的邮件内容
		word_list = text_parse(codecs.open("../machine_learning_inaction_sourcecode/Ch04/email/spam/%d.txt" % i, 
								"r", encoding = "utf-8", errors='ignore').read())
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(1)
		word_list = text_parse(codecs.open("../machine_learning_inaction_sourcecode/Ch04/email/ham/%d.txt" % i,
								"r", encoding = "utf-8", errors='ignore').read())
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(0)
	vocablist = create_vocablist(doc_list)

	# 从中随机构建训练数据和测试数据
	trainSet = list(range(50))
	testSet = []
	for i in range(10):
		index = int(random.uniform(0,len(trainSet)))
		testSet.append(index)
		del(trainSet[index])
	train_matrix = []
	train_category = []
	for index in trainSet:
		train_matrix.append(bag_of_words2vec(vocablist, doc_list[index]))
		train_category.append(class_list[index])
	# 开始训练
	p0,p1,pAb = train(train_matrix, train_category)
	# 统计识别率
	error_count = 0
	for doc_index in testSet:
		word_vector = bag_of_words2vec(vocablist, doc_list[doc_index])
		if classify(word_vector, p0,p1,pAb) != class_list[doc_index]:
			error_count += 1
	print("the error rate is :", float(error_count) / len(testSet))


def cal_most_freq(vocablist, full_text):
	"""
	@brief      计算出现频率
	@param      vocablist  The vocablist
	@param      full_text  The full text
	@return     { description_of_the_return_value }
	"""
	freq_dict = dict()
	for token in vocablist:
		freq_dict[token] = full_text.count(token)
	sorted_freq = sorted(freq_dict.items(), key = lambda x : x[1],reverse = True)
	return sorted_freq[:30]


def local_words(feed1, feed0):
	"""
	@brief      基于贝叶斯的广告区域倾向分类器
	@param      feed1  The feed 1
	@param      feed0  The feed 0
	@return     { description_of_the_return_value }
	"""
	doc_list = list()
	class_list = list()
	full_text = list()

	min_len = min(len(feed1["entries"]), len(feed0["entries"]))
	for i in range(min_len):
		word_list = text_parse(feed1["entries"][i]["summary"])
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(1)
		word_list = text_parse(feed0["entries"][i]["summary"])
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(0)

	vocablist = create_vocablist(doc_list)
	# 去除频率最高的前30个单词
	top30_words = cal_most_freq(vocablist, full_text)
	for pair_word in top30_words:
		if pair_word[0] in vocablist:
			vocablist.remove(pair_word[0])

	trainSet = list(range(2 * min_len))
	print(len(trainSet))
	testSet = list()
	for i in range(20):
		index = int(random.uniform(0,len(trainSet)))
		print(index)
		testSet.append(trainSet[index])
		del(trainSet[index])

	train_matrix = list()
	train_category = list()
	for doc_index in trainSet:
		train_matrix.append(bag_of_words2vec(vocablist, doc_list[doc_index]))
		train_categorya.append(class_list[doc_index])

	p0, p1, pSpam = train(train_matrix, train_category)

	error_count = 0
	for doc_index in testSet:
		word_vector = bag_of_words2vec(vocablist, doc_list[doc_index])
		if classify(word_vector, p0,p1,pAb) != class_list[doc_index]:
			error_count += 1
	print("the error rate is :", float(error_count) / len(testSet))
	return vocablist, p0, p1

if __name__ == '__main__':
	test()
	spam_test()
	feed1 = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
	print (feed1)
	feed0 = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")
	local_words(feed1, feed0)



