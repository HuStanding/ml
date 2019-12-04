# -*- coding: utf-8 -*-
'''
@Author: huzhu
@Date: 2019-11-29 15:44:23
@Description: 
'''
import numpy as np
import graphviz
import matplotlib.pyplot as plt

def plot_decision(model, x_data, y_data):
    x_data = np.mat(x_data)
    y_data = np.mat(y_data).T
    x_train, y_train = x_data, y_data
    clf = model.fit(x_train, y_train)
    #绘制决策树
    dot_data=tree.export_graphviz(clf,out_file=None,filled=True,rounded=True,special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()
    result = model.predict(x_train)
    plt.figure(figsize=(8,4))
    srt_idx = x_train.argsort(0)
    plt.plot(x_train[srt_idx].reshape(-1,1), y_train[srt_idx].reshape(-1,1), 'go', label = "true value")
    plt.plot(x_train[srt_idx].reshape(-1,1), result[srt_idx].reshape(-1,1), 'ro-', label = "predict value")
    plt.title("decision tree regression")
    plt.legend()
    plt.show()

def get_params(x_data, y_data):
    x_data = np.mat(x_data)
    y_data = np.mat(y_data).T
    s=[1.5,2.5,3.5,4.5,5.5]
    min_ms = np.inf
    left = list()
    right = list()
    c1_list = list()
    c2_list = list()
    for i in s:
        left = np.where(x_data < i)[0]
        right = np.where(x_data > i)[0]
        c1_list = [y_data[k] for k in left]
        c2_list = [y_data[k] for k in right]
        c1 = np.mean(c1_list, axis = 0)
        c2 = np.mean(c2_list, axis = 0)
        sm = 0
        for j in left:
            sm += (y_data[j] - c1) ** 2
        for j in right:
            sm += (y_data[j] - c2) ** 2
        if sm < min_ms:
            min_ms = sm
        print(i, sm)

        


if __name__ == "__main__":
    x_data = [[1],[2],[3],[4],[5],[6]]
    y_data = [5.56, 5.70, 5.91, 6.40,6.90,7.95]
    # 决策树回归
    from sklearn import tree
    get_params(x_data,y_data)
    model_decisiontree_regression = tree.DecisionTreeRegressor(min_impurity_decrease=0.01)
    #plot_decision(model_decisiontree_regression, x_data, y_data)
    