# -*- coding: utf-8 -*-
'''
@FilePath: /nlp-exercise/fasttext/demo_02.py
@Author: huzhu
@Date: 2020-04-19 16:32:22
@Description: 文本分类模型
'''
import fasttext
#help(fasttext.train_supervised)
#model = fasttext.train_supervised(input="data/cooking.train", epoch=25)
#model.save_model("model_cooking.bin")

model= fasttext.load_model("model_cooking.bin")
print(model.predict("Which baking dish is best to bake a banana bread ?"))
print(model.predict("Why not put knives in the dishwasher?"))
print(model.test("data/cooking.train"))
print(model.test("data/cooking.valid"))