# -*- coding: utf-8 -*-
'''
@Author: huzhu
@Date: 2020-01-09 14:03:46
@Description: 
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler


def load_dataset(file_path):
    dataMat = []
    labelMat = []
    fr = open(file_path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return np.array(dataMat), np.array(labelMat).reshape((-1,1))


def nn_lib():
    train_data_arr, train_label_arr = load_dataset('train_dataset.txt')
    test_data_arr, test_label_arr = load_dataset('test_dataset.txt')

    scaler = StandardScaler() # 标准化转换
    scaler.fit(train_data_arr)  # 训练标准化对象
    train_data_arr = scaler.transform(train_data_arr)   # 转换数据集
    scaler.fit(test_data_arr)  # 训练标准化对象
    test_data_arr = scaler.transform(test_data_arr)   # 转换数据集

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,), random_state=1)
    clf.fit(train_data_arr, train_label_arr)
    print('每层网络层系数矩阵维度：\n',[coef.shape for coef in clf.coefs_])
    cengindex = 0
    for wi in clf.coefs_:
        cengindex += 1  # 表示底第几层神经网络。
        print('第%d层网络层:' % cengindex)
        print('权重矩阵维度:',wi.shape)
        print('系数矩阵:\n',wi)

    r = clf.score(train_data_arr, train_label_arr)
    print("R值(准确率):", r)

    y_predict = clf.predict(test_data_arr).reshape((-1,1))
    accurcy = np.sum(y_predict == test_label_arr) / len(test_data_arr)
    print(accurcy)

    X = test_data_arr
    y = test_label_arr
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(y))
    plt.show()


if __name__ == "__main__":
    nn_lib()