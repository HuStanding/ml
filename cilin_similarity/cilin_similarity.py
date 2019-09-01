# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-08-27 09:23:51
# @Last Modified by:   huzhu
# @Last Modified time: 2019-09-01 15:40:56
import math
import codecs


class SilinSimilarity(object):
    """docstring for SilinSimilarity"""

    def __init__(self, cilin_path):
        """
        @brief      init
        @param      self  The object
        @param      cilin_path  词林文件路径
        @return     No
        """
        # 各层的系数
        self.a = 0.65
        self.b = 0.8
        self.c = 0.9
        self.d = 0.96
        self.e = 0.5
        self.f = 0.1

        self.code_words = dict()  # key为编码，words为对应的词列表
        self.word_codes = dict()  # key为词，codes为对应的编码列表

        self.load_data(cilin_path)

    def load_data(self, path):
        """
        @brief      Loads a data.
        @param      self  The object
        @param      path  The cilin path
        @return     No
        """
        with codecs.open(path, "r", encoding="gbk") as f:
            for line in f.readlines():
                item = line.strip().split(" ")
                code = item[0]
                words = item[1:]
                self.code_words[code] = words
                for word in words:
                    if word in self.word_codes.keys():
                        self.word_codes[word].append(code)
                    else:
                        self.word_codes[word] = [code]

    def get_similarity(self, word1, word2):
        """
        @brief      Gets the similarity.
        @param      self   The object
        @param      word1  The word 1
        @param      word2  The word 2
        @return     The similarity.
        """
        code1_lst = self.word_codes.get(word1, None)
        code2_lst = self.word_codes.get(word2, None)
        # 取最大的相似度作为结果
        max_sim = 0
        for code1 in code1_lst:
            for code2 in code2_lst:
                sim = round(self.get_sim(code1, code2), 4)
                #print(code1, code2, "相似度为 ", sim)
                max_sim = max(sim, max_sim)
        return max_sim

    def get_sim(self, code1, code2):
        """
        @brief      Gets the simulation.
        @param      self   The object
        @param      code1  The code 1
        @param      code2  The code 2
        @return     相似度
        """
        common_level = self.get_common_level(code1, code2)
        n = self.get_n(code1, code2)
        k = self.get_k(code1, code2)
        # 以@结尾的
        if code1 == code2:
            if code1[-1] == "=":  # 结尾是"="表示是同义词
                return 1
            else:   # 结尾不可能为"@"，只能是"#"
                return self.e
        if code1.endswith("@") or code2.endswith("@") or common_level == 0:
            return self.f
        else:
            if common_level == 2:
                return self.cal_sim(self.a, n, k)
            elif common_level == 3:
                return self.cal_sim(self.b, n, k)
            elif common_level == 4:
                return self.cal_sim(self.c, n, k)
            else:
                return self.cal_sim(self.d, n, k)

    def get_common_level(self, code1, code2):
        """
        @brief      Gets the common level.
        @param      self   The object
        @param      code1  The code 1
        @param      code2  The code 2
        @return     The common level.
        """
        common_code = self.get_common_code(code1, code2)
        if len(common_code) == 1:
            return 2
        elif len(common_code) == 2:
            return 3
        elif len(common_code) == 4:
            return 4
        elif len(common_code) == 5:
            return 5
        else:
            return 0

    def get_n(self, code1, code2):
        """
        @brief      获取分支层的所有节点数
        @param      self          The object
        @param      common_level  The common layer
        @param      n             
        @param      k             
        @return     {}
        """
        common_code = self.get_common_code(code1,code2)
        common_level = self.get_common_level(code1, code2)
        if len(common_code) == 0:
        	return 0
        res = set()
        for key in self.code_words.keys():
            if key.startswith(common_code):
                code = self.split_code(key)
                res.add(code[common_level - 1])
        return len(res)

    def get_k(self,code1,code2):
        """
        @brief      获取分支之间的距离
        @param      self   The object
        @param      code1  The code 1
        @param      code2  The code 2
        @return     The k.
        """
        code1 = self.split_code(code1)
        code2 = self.split_code(code2)
        if code1[0] != code2[0]:
            return abs(ord(code1[0]) - ord(code2[0]))
        elif code1[1] != code2[1]:
            return abs(ord(code1[1]) - ord(code2[1]))
        elif code1[2] != code2[2]:
            return abs(int(code1[2]) - int(code2[2]))
        elif code1[3] != code2[3]:
            return abs(ord(code1[3]) - ord(code2[3]))
        else:
            return abs(int(code1[4]) - int(code2[4]))

    def cal_sim(self, coeff, n, k):
        """
        @brief      计算相似度
        @param      self          The object
        @param      common_level  The common layer
        @param      n             
        @param      k             
        @return     相似度值
        """
        return coeff * math.cos(n * math.pi / 180) * ((n - k + 1) / n)

    def get_common_code(self, code1, code2):
        """
        @brief      获取公共子串
        @param      self   The object
        @param      code1  The code 1
        @param      code2  The code 2
        @return     { description_of_the_return_value }
        """
        code1 = self.split_code(code1)
        code2 = self.split_code(code2)
        res = "" 
        for i, j in zip(code1, code2):
            if i == j:
                res += i
            else:
                break
        return res

    def split_code(self,code): 
        """
        @brief      将code进行编码划分
        @param      self   The object
        @param      code1  The code
        @return     { description_of_the_return_value }
        """
        code = [code[0], code[1], code[2:4],
                 code[4], code[5:7], code[-1]]
        return code



if __name__ == '__main__':
    test = SilinSimilarity("data/cilin.txt")
    word1 = "骄傲"
    word2 = "仔细"
    print(word1, word2, "相似度为", test.get_similarity(word1, word2))
    word1 = "人民"
    word2 = "国民"
    print(word1, word2, "相似度为", test.get_similarity(word1, word2))
    word1 = "人民"
    word2 = "群众"
    print(word1, word2, "相似度为", test.get_similarity(word1, word2))
    word1 = "人民"
    word2 = "同志"
    print(word1, word2, "相似度为", test.get_similarity(word1, word2))
    word1 = "人民"
    word2 = "良民"
    print(word1, word2, "相似度为", test.get_similarity(word1, word2))
    word1 = "人民"
    word2 = "先锋"
    print(word1, word2, "相似度为", test.get_similarity(word1, word2))