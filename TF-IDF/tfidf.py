# -*- coding: utf-8 -*-
'''
@Author: huzhu
@Date: 2019-12-13 09:36:29
@Description: TF-IDF 实现
'''

from collections import Counter
import math

def tf(word, count):
    return count[word] / sum(count.values())


def idf(word, count_list):
    n_contain = sum([1 for count in count_list if word in count])
    return math.log(len(count_list) / (1 + n_contain))


def tf_idf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


if __name__ == "__main__":
    # 语料库
    corpus = ['this is the first document',
            'this is the second second document',
            'and the third one',
            'is this the first document']
    words_list = list()
    for i in range(len(corpus)):
        words_list.append(corpus[i].split(' '))
    print(words_list)

    # 统计词语数量
    count_list = list()
    for i in range(len(words_list)):
        count = Counter(words_list[i])
        count_list.append(count)
    print(count_list)

    # 输出结果
    for i, count in enumerate(count_list):
        print("第 {} 个文档 TF-IDF 统计信息".format(i + 1))
        scores = {word : tf_idf(word, count, count_list) for word in count}
        sorted_word = sorted(scores.items(), key = lambda x : x[1], reverse=True)
        for word, score in sorted_word:
            print("\tword: {}, TF-IDF: {}".format(word, round(score, 5)))
    



    
