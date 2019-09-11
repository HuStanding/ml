# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-09-11 14:59:12
# @Last Modified by:   huzhu
# @Last Modified time: 2019-09-11 16:53:30

from pyltp import NamedEntityRecognizer
from pyltp import Segmentor
from pyltp import Postagger
LTP_DATA_DIR = '/Users/hz/Documents/codes/nlp/ltp_data_v3.4.0/'


def segmentor(sentence='你好，你觉得这个例子从哪里来的？当然还是直接复制官方文档，然后改了下这里得到的。'):
	"""
	@brief      { 分词 }
	@param      sentence  The sentence
	@return     { description_of_the_return_value }
	"""
	segmentor = Segmentor()  # 初始化实例
	segmentor.load(LTP_DATA_DIR + "cws.model")  # 加载模型
	words = segmentor.segment(sentence)  # 分词
	# 可以转换成List 输出
	words_list = list(words)
	print(" ".join(words_list))
	segmentor.release()  # 释放模型
	return words_list

def ner(words, postags):
	"""
	@brief      { ner识别 }
	@param      words    The words
	@param      postags  The postags
	@return     { 实体类型 }
	"""
	recognizer = NamedEntityRecognizer() # 初始化实例
	recognizer.load(LTP_DATA_DIR + "ner.model")  # 加载模型
	netags = recognizer.recognize(words, postags)  # 命名实体识别
	for word, ntag in zip(words, netags):
	    print(word + '/' + ntag)
	recognizer.release()  # 释放模型
	return netags

def posttagger(words):
	"""
	@brief      { 词性标注 }
	@param      words  The words
	@return     { description_of_the_return_value }
	"""
	postagger = Postagger() # 初始化实例
	postagger.load(LTP_DATA_DIR + "pos.model")  # 加载模型
	postags = postagger.postag(words)  # 词性标注
	# for word,tag in zip(words,postags):
	#     print(word+'/'+tag)
	postagger.release()  # 释放模型
	return postags

if __name__ == '__main__':
	# 测试分词
	words = segmentor('我家在安徽，我现在在北京工作。王者荣耀的英雄李白有两段位移技能。')
	# 测试标注
	tags = posttagger(words)
	# 命名实体识别
	ner(words,tags)

