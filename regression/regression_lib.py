# -*- coding: utf-8 -*-
'''
@Author: huzhu
@Date: 2019-11-26 21:55:53
@Description: 使用sklearn包进行回归分析，参考：https://blog.csdn.net/Yeoman92/article/details/75051848
'''
from regression import load_data
import numpy as np
import matplotlib.pyplot as plt
import graphviz

def plot_regression(model, x_data, y_data):
    x_data = np.mat(x_data)
    y_data = np.mat(y_data).T
    x_train, y_train = x_data[:150,1:], y_data[:150,:]
    x_test, y_test = x_data[150:,1:], y_data[150:,:]
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    train_result = model.predict(x_train)
    # 绘制
    plt.figure(figsize=(8,4))
    srt_idx = x_train.argsort(0)
    plt.plot(x_train[srt_idx].reshape(-1,1), y_train[srt_idx].reshape(-1,1), 'go', label = "trian data")
    plt.plot(x_train[srt_idx].reshape(-1,1), train_result[srt_idx].reshape(-1,1), 'r-', label = "regression value")
    plt.title("score:%f" % score)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    x_data, y_data = load_data("ex0.txt")

    # 线性回归
    from sklearn import linear_model
    model_linear_regression = linear_model.LinearRegression()

    # 决策树回归
    from sklearn import tree
    model_decisiontree_regression = tree.DecisionTreeRegressor(min_weight_fraction_leaf=0.01)

    # SVM回归
    from sklearn import svm
    model_svr = svm.SVR()

    # KNN回归
    from sklearn import neighbors 
    model_knn = neighbors.KNeighborsRegressor()

    # 随机森林回归
    from sklearn import ensemble
    model_random_forest = ensemble.RandomForestRegressor(n_estimators=20)

    # AdaBoost回归
    from sklearn import ensemble
    model_adaboost = ensemble.AdaBoostRegressor(n_estimators=50)

    # GBRT回归
    from sklearn import ensemble
    model_gradient_boost = ensemble.GradientBoostingRegressor(n_estimators=100)

    # Bagging回归
    from sklearn.ensemble import BaggingRegressor
    model_bagging = BaggingRegressor()

    # ExtraTree极端随机树回归
    from sklearn.tree import ExtraTreeRegressor
    model_extratree = ExtraTreeRegressor()

    # Ridge回归
    model_ridge = linear_model.Ridge(alpha = 0.01)

    # 绘制回归曲线
    plot_regression(model_svr, x_data, y_data)
    #plot_decision(model_decisiontree_regression, x_data,y_data)
    

