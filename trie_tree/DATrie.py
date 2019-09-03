# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-08-25 15:28:24
# @Last Modified by:   huzhu
# @Last Modified time: 2019-09-03 09:45:25
import numpy as np

ARRAY_SIZE = 655350
BASE_ROOT = 1
BASE_NULL = 0
CHECK_ROOT = -1
CHECK_NULL = -2


class TrieNode(object):
    """docstring for TrieNode"""

    def __init__(self):
        self.transfor_ratio = BASE_NULL  # 转移基数
        self.isLeaf = False  # 是否为叶子节点
        self.label = 0  # 节点标志
        self.value = -1  # 当该节点为叶子节点时关联的字典表中对应词条的索引号

        
class DATrie(object):
    """docstring for DATrie"""

    def __init__(self):
        self.base = [TrieNode() for i in range(ARRAY_SIZE)]
        self.check = np.zeros(ARRAY_SIZE)-2
        self.root = TrieNode()
        self.root.transfor_ratio = BASE_ROOT
        self.base[0] = self.root
        self.check[0] = CHECK_ROOT

    def build(self, words):
        pos = 0
        # 每次都从单词的第i个开始插入，减少冲突
        for j in range(len(words)):
            for index in range(len(words)):
                chars = list(words[index])
                if len(chars) > pos:
                    start_state = 0
                    for i in range(0, pos):
                        start_state = self.transfor_ratio(
                            start_state, self.get_code(chars[i]))

                    node = self.insert(start_state, self.get_code(
                        chars[pos]), len(chars) == pos + 1, index)
                    node.label = chars[pos]
            pos += 1

    def match(self, keyword):
        res = []
        chars = list(keyword)
        for i in range(len(keyword)):
            start_state = 0
            for j in range(len(keyword)):
                end_state = self.transfor_ratio(
                    start_state, self.get_code(chars[j]))
                if self.base[end_state].transfor_ratio != BASE_NULL and self.check[end_state] == start_state:
                    if self.base[end_state].isLeaf:
                        if self.base[end_state].value not in res:
                            res.append(self.base[end_state].value)
                    start_state = end_state
                else:
                    break
        return res

    def start_with(self, keyword):
        chars = list(keyword)
        start_state = 0
        for j in range(len(keyword)):
            end_state = self.transfor_ratio(
                start_state, self.get_code(chars[j]))
            if self.base[end_state].transfor_ratio != BASE_NULL and self.check[end_state] == start_state:
                start_state = end_state
                continue
            else:
                return False
        return True

    def print_trie(self):
        print()
        print("%-7s" % "index", end="")
        base = self.base
        for i in range(ARRAY_SIZE):
            if self.base[i].transfor_ratio != BASE_NULL:
                print("%-7d" % i, end="")

        print()
        print("%-7s" % "leaf", end="")
        for i in range(ARRAY_SIZE):
            if self.base[i].transfor_ratio != BASE_NULL:
                print("%-7d" % base[i].isLeaf, end="")

        print()
        print("%-7s" % "ratio", end="")
        for i in range(ARRAY_SIZE):
            if self.base[i].transfor_ratio != BASE_NULL:
                print("%-7d" % base[i].transfor_ratio, end="")

        print()
        print("%-7s" % "value", end="")
        for i in range(ARRAY_SIZE):
            if self.base[i].transfor_ratio != BASE_NULL:
                print("%-7d" % base[i].value, end="")

        print()
        print("%-7s" % "char", end="")
        for i in range(1, ARRAY_SIZE):
            if self.base[i].transfor_ratio != BASE_NULL:
                print("%-7s" % base[i].label, end="")

        print()
        print("%-7s" % "check", end="")
        for i in range(ARRAY_SIZE):
            if self.base[i].transfor_ratio != BASE_NULL:
                print("%-7d" % self.check[i], end="")

        print()

    def insert(self,  start_state, offset, isLeaf, index):
        end_state = self.transfor_ratio(start_state, offset)  # 状态转移
        base = self.base
        check = self.check
        if (base[end_state].transfor_ratio != BASE_NULL) and (check[end_state] != start_state):
            # 已被占用
            while base[end_state].transfor_ratio != BASE_NULL:
                end_state += 1
            base[start_state].transfor_ratio = (
                end_state - offset)  # 改变父节点转移基数

        if isLeaf:
            base[end_state].transfor_ratio = abs(
                base[start_state].transfor_ratio)*-1  # 叶子节点转移基数标识为父节点转移基数的相反数
            base[end_state].isLeaf = True
            base[end_state].value = index  # 为叶子节点时需要记录下该词在字典中的索引号
        else:
            if base[end_state].transfor_ratio == BASE_NULL:  # 未有节点经过
                base[end_state].transfor_ratio = abs(
                    base[start_state].transfor_ratio)  # 非叶子节点的转移基数一定为正

        check[end_state] = start_state  # check中记录当前状态的父状态

        return base[end_state]

    def transfor_ratio(self, start_state, offset):
        return abs(self.base[start_state].transfor_ratio)+offset  # 状态转移

    def get_code(self, c):
        return ord(c)


if __name__ == '__main__':
    words = ["清华", "清华大学", "清新", "中华", "中华人民", "华人","CSDN","Python","Python3","Java","贝壳找房"]
    dat = DATrie()
    dat.build(words)
    dat.print_trie()
    print(dat.match("清"))
    print(dat.match("Python"))
    print(dat.match("中华人民共和国"))
    print(dat.match("贝壳找房"))
    print(dat.start_with("贝壳"))
    print(dat.start_with("Pytho"))
    print(dat.start_with("中"))

