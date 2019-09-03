# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-09-02 20:31:28
# @Last Modified by:   huzhu
# @Last Modified time: 2019-09-03 09:56:15
import DATrie

def segment(tree, text):
	words = list()
	flag = 0
	delta = 1
	while flag + delta <= len(text):
		temp = text[flag:flag + delta + 1]
		if tree.start_with(temp):
			if flag + delta == len(text):
				words.append(temp)
				break
			delta += 1
			continue
		words.append(temp[0:-1])
		flag = flag + delta
		delta = 1
	return words


if __name__ == '__main__':
	words = ["中国", "人名", "人民", "孙健", "CSDN", "java",
	    "java学习", "部分"]
	tree = DATrie.DATrie()
	tree.build(words)
	text = "中国人名识别是中国人民的一个骄傲.孙健人民在CSDN中学到了很多最早iteye是java学习笔记叫javaeye但是java123只是一部分"
	print(segment(tree,text))
