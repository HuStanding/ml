# -*- coding: utf-8 -*-
'''
FilePath: /ml/word2vec/w2v_demo01.py
Author: huzhu
Date: 2020-09-13 12:35:35
Description: word2vec python源码版本 demo01
'''
import re
import pandas as pd
import numpy as np 
from random import sample


def load_data(path):
    df = pd.read_csv(path, sep=",")
    review = df["review"].tolist()[1:2]
    review = [eval(x) for x in review]
    return review
    
def sigmoid(x):
    ### YOUR CODE HERE (~1 Line)
    s = np.exp(x)/(1+np.exp(x))
    ### END YOUR CODE

    return s
class Word2VecDemo:
    
    def __init__(self):
        self.corpus = []
        self.vocab = []
        self.theta = []
        self.WORD_NUM = 0
        self.WINDOW_SIZE = 2
    
    def build_vocab(self, data):
        res = set()
        for item in data:
            res |= set(item)
        return list(res)
    
    def get_back_word(self, center_word, data):
        res = {}
        for item in data:
            if center_word not in item:
                continue
            if len(item) < 2:
                continue
            index = item.index(center_word)
            for i in range(-self.WINDOW_SIZE, self.WINDOW_SIZE + 1):
                back_word_index = index + i
                if back_word_index < 0 or back_word_index == index or back_word_index >= len(item):
                    continue
                back_word = item[back_word_index]
                res[back_word] = res.get(back_word, 0) + 1
        return res

    def predict(self, word):
        index = self.vocab[word]
        return self.vocab[index]
    
    def cal_proba(self, u_o, v_c):
        res = np.exp(u_o.dot(v_c))
        tmp = 0
        for item in self.theta[self.WORD_NUM:]:
            tmp += np.exp(item.dot(v_c))
        res = res / tmp
        return res

    def cal_loss(self, data):
        loss = 0
        for center_word in self.vocab:
            back_words = self.get_back_word(center_word, data)
            v_c, _ = self.get_word_vector(center_word)
            for back_word, num in back_words.items():
                _, u_o = self.get_word_vector(back_word)
                loss += num * np.log(self.cal_proba(u_o, v_c))
            loss = -loss / self.WORD_NUM
        return loss

    def get_word_vector(self, word):
        index = self.vocab.index(word)
        # 返回中心词向量，背景词向量
        return self.theta[index], self.theta[index + self.WORD_NUM]
    
    def naive_softmax(self, data, alpha):
        # 随机梯度下降
        for index in range(self.WORD_NUM):
            word = self.vocab[index]
            v_c, _ = self.get_word_vector(word)
            back_words = self.get_back_word(word, data)
            delta_center = 0
            total_back_num = 0
            for back_word, num in back_words.items():
                _, u_o = self.get_word_vector(back_word)
                delta_center += u_o * num
                total_back_num += num
            # 计算中心词的梯度
            tmp = 0
            for j in range(self.WORD_NUM):
                item_word = self.vocab[j]
                _, u_j = self.get_word_vector(item_word)
                tmp += self.cal_proba(u_j, v_c) * u_j
            delta_center = delta_center - tmp * total_back_num
            self.theta[index] += alpha * delta_center

        for index in range(self.WORD_NUM):
            word = self.vocab[index]
            v_c, _ = self.get_word_vector(word)
            # 计算所有背景词的梯度
            for center_word, other_back_words in self.back_word_map.items():
                if center_word == word:
                    # 这里的所有背景词向量使用该梯度更新
                    for tmp in other_back_words:
                        _, u_o = self.get_word_vector(tmp)
                        delta_back = (v_c - self.cal_proba(u_o, v_c) * v_c) * other_back_words[tmp]
                        back_index = self.vocab.index(tmp) + self.WORD_NUM
                        self.theta[back_index] += alpha * delta_back
                else:
                    # 这里的所有背景词向量使用该梯度更新
                    for tmp in other_back_words:
                        _, u_o = self.get_word_vector(tmp)
                        delta_back = -self.cal_proba(u_o, v_c) * v_c * other_back_words[tmp]
                        back_index = self.vocab.index(tmp) + self.WORD_NUM
                        self.theta[back_index] += alpha * delta_back
    

    def get_random_words(self, back_words, K):
        tmp = [x for x in self.vocab if x not in back_words] 
        return sample(tmp, 5)

    def neg_sampling(self, data, alpha, K=5):
        # 随机梯度下降
        for index in range(self.WORD_NUM):
            word = self.vocab[index]
            v_c, _ = self.get_word_vector(word)
            back_words = self.get_back_word(word, data)
            # 随机的其他背景词
            random_back_words = self.get_random_words(list(back_words.keys()), K)
            delta_center = 0
            for back_word, num in back_words.items():
                _, u_o = self.get_word_vector(back_word)
                delta_center  = -(1 - sigmoid(u_o.dot(v_c))) * u_o
                for back_word in random_back_words:
                    _, u_k = self.get_word_vector(back_word)
                    delta_center += (1 - sigmoid(-u_k.dot(v_c))) * u_k
            # 计算中心词的梯度
            self.theta[index] -= alpha * delta_center
            # 对上下文单词求导
            for back_word in back_words:
                back_index = self.vocab.index(back_word) + self.WORD_NUM
                _, u_o = self.get_word_vector(back_word)
                delta_back = -(1 - sigmoid(u_o.dot(v_c)))*v_c
                self.theta[back_index] -= alpha * delta_back
            # 对负采样的背景词求导
            for back_word in random_back_words:
                back_index = self.vocab.index(back_word)+ self.WORD_NUM
                _, u_k = self.get_word_vector(back_word)
                delta_back = (1 - sigmoid(-u_k.dot(v_c)))*v_c
                self.theta[back_index] -= alpha * delta_back

    def skip_gram(self, data, iters=1000, m=2, alpha=0.1, dim=10):
        # 建立词典
        self.WINDOW_SIZE = m
        self.vocab = self.build_vocab(data)
        self.WORD_NUM = len(self.vocab)
        # 建立背景词映射关系
        self.back_word_map = {}
        for word in self.vocab:
            self.back_word_map[word] = self.get_back_word(word, data)
        rdm = np.random.RandomState(1) 
        # 词向量矩阵
        self.theta = rdm.rand(self.WORD_NUM, dim)
        self.theta = np.vstack((self.theta,self.theta))
        for item_num in range(iters):
            # 计算每次迭代的时候的损失函数值
            loss = self.cal_loss(data)
            if item_num % 10 == 0:
                print("After {} iteration, loss is {}".format(item_num, loss))
                #self.naive_softmax(data, alpha)
                self.neg_sampling(data, alpha, K=5)
            
            


if __name__ == "__main__":
    #inpath = 'data/movie_data.csv'
    #train_data = load_data(inpath)
    train_data = [
        ["I", "like", "NLP"],
        ["I", "enjoy", "computer", "games", "and", "machine", "learning"],
        ["I", "like", "deep", "learning"]
    ]
    test = Word2VecDemo()
    test.skip_gram(train_data, iters=500, m=2, alpha=0.01, dim=100)