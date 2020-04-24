# -*- coding: utf-8 -*-
'''
@FilePath: /nlp-exercise/fasttext/demo_01.py
@Author: huzhu
@Date: 2019-11-17 18:23:43
@Description: 
'''
from gensim.models import FastText
sentences = [["你", "是", "谁"], ["我", "是", "中国人"]]

model = FastText(sentences,  model = 'cbow', size=4, window=3, min_count=1, iter=10,min_n = 3 , max_n = 6,word_ngrams = 0)
print(model['你'])  # 词向量获得的方式
