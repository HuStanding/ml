# -*- coding: utf-8 -*-
'''
@FilePath: /ML/boosting_tree/cart.py
@Author: huzhu
@Date: 2020-05-31 18:08:15
@Description: CART 树回归
'''

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset,label = load_boston(return_X_y=True) # 波士顿的房价数据集
# 将 label 插入到 dataset 的最后一列
dataset = np.insert(dataset, 0, values=label, axis=1)

def binsplit_dataset(dataset, feature, value):
    mat0 = dataset[np.nonzero(dataset[:,feature] > value)[0],:]
    mat1 = dataset[np.nonzero(dataset[:,feature] <= value)[0],:]
    return mat0, mat1


def reg_leaf(dataset):
    return np.mean(dataset[:,-1])


def reg_err(dataset):
    return np.var(dataset[:,-1]) * np.shape(dataset)[0]


def create_tree(dataset, leaf_type=reg_leaf, err_type=reg_err,ops=(1,4)):
    feat, val = choose_best_split(dataset, leaf_type, err_type, ops)
    if feat == None:
        return val
    ret_tree = {}
    ret_tree["sp_ind"] = feat
    ret_tree["sp_val"] = val
    lset, rset = binsplit_dataset(dataset, feat, val)
    ret_tree["left"] = create_tree(lset, leaf_type, err_type, ops)
    ret_tree["right"] = create_tree(rset, leaf_type, err_type, ops)
    return ret_tree


def choose_best_split(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)):
    # 误差下降值，最小样本数
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataset[:,-1].T.tolist())) == 1:
        return None, leaf_type(dataset)
    m ,n = np.shape(dataset)
    S = err_type(dataset)
    bestS = np.inf; beatindex = 0; bestvalue = 0
    for feat_index in range(n - 1):
        for split_val in set(dataset[:, feat_index]):
            mat0, mat1 = binsplit_dataset(dataset, feat_index, split_val)
            # 如果发现切分的样本数太少
            if(np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN):
                continue
            newS = err_type(mat0) + err_type(mat1)
            if newS < bestS:
                beatindex = feat_index
                bestvalue = split_val
                bestS = newS
    # 如果误差减小不大则退出
    if (S - bestS) < tolS:
        return None, leaf_type(dataset)
    mat0, mat1 = binsplit_dataset(dataset, beatindex, bestvalue)
    # 如果切分的数据集很小则退出
    if (np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN):
        return None, leaf_type(dataset)
    return beatindex, bestvalue


if __name__ == "__main__":
    tree = create_tree(dataset,leaf_type=reg_leaf, err_type=reg_err,ops=(1,50))
    print(tree)