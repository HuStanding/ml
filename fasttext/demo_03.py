# -*- coding: utf-8 -*-
'''
@FilePath: /nlp-exercise/fasttext/demo_03.py
@Author: huzhu
@Date: 2020-04-23 22:01:50
@Description: word 表示
'''
import fasttext
#model = fasttext.train_unsupervised('data/fil9')
#model.save_model("result/fil9.bin")
model = fasttext.load_model("result/fil9.bin")
print(len(model.words))
print(model.get_word_vector("the"))
print(model.get_nearest_neighbors('hello'))
print(model.get_nearest_neighbors('enviroment'))
print(model.get_analogies("berlin", "germany", "china"))
print(model.get_nearest_neighbors('beijing'))