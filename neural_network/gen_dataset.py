# -*- coding: utf-8 -*-
'''
@Author: huzhu
@Date: 2020-01-09 18:06:36
@Description: 加载数据集
'''
import numpy as np
import matplotlib.pyplot as plt

def gen_dataset():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    # np.random.seed(0)
    # X, y = datasets.make_moons(200, noise=0.20)
    np.random.seed(13)
    X, y = make_blobs(centers=4, n_samples = 200)
    # 绘制数据分布
    plt.figure(figsize=(6,4))
    plt.scatter(X[:,0], X[:,1],c=y)
    plt.title("Dataset")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.show()

    # 重塑目标以获得具有 (n_samples, 1)形状的列向量
    y = y.reshape((-1,1))
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train_dataset = np.append(X_train,y_train, axis = 1)
    test_dataset = np.append(X_test,y_test, axis = 1)
    np.savetxt("train_dataset.txt", train_dataset, fmt="%.4f %.4f %d")
    np.savetxt("test_dataset.txt", test_dataset, fmt="%.4f %.4f %d")


if __name__ == "__main__":
    gen_dataset()