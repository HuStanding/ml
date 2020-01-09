# -*- coding: utf-8 -*-
'''
@Author: huzhu
@Date: 2019-12-02 10:02:32
@Description: Logistic回归
'''

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatrix, classLabels):
    dataMatrix = mat(dataMatrix)  # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    m, n = shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = ones((n, 1))
    weights_list = list()
    for k in range(maxCycles):  # heavy on matrix operations
        h = sigmoid(dataMatrix*weights)  # matrix mult
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
        weights_list.append(weights)
    return weights, weights_list

def stocGradAscent(dataMatrix, classLabels, numIter=150):
    dataMatrix = mat(dataMatrix)
    labelMat = mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)                                                
    weights = np.ones((n,1))  
    weights_list = list()                                                    
    for j in range(numIter):                                           
        dataIndex = list(range(m))
        for i in range(m):           
            alpha = 4/(1.0+j+i)+0.01            
            randIndex = int(random.uniform(0,len(dataIndex)))              
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                   
            error = classLabels[randIndex] - h                               
            weights = weights + alpha * dataMatrix[randIndex].transpose() * error 
            del(dataIndex[randIndex])  
            weights_list.append(weights)                                      
    return weights, weights_list                                              


def logistic_lib(dataMatrix, classLabels):
    '''
    @description: 使用 sklearn 包调用 logistic 回归函数
    @param {type} 
    @return: 
    '''
    from sklearn import linear_model
    model_logistic_regression = linear_model.LogisticRegression(solver='liblinear',max_iter=10)
    classifier = model_logistic_regression.fit(dataMatrix, classLabels)
    print(model_logistic_regression.coef_)
    print(model_logistic_regression.intercept_)
    accurcy = classifier.score(dataMatrix, classLabels) * 100
    print(accurcy)


    
def plot_weights(weights_list):
    font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=12)
    fig = plt.figure(figsize=(8, 8))
    x = range(len(weights_list))
    w0 = [item[0, 0] for item in weights_list]
    w1 = [item[1, 0] for item in weights_list]
    w2 = [item[2, 0] for item in weights_list]
    plt.subplot(311)
    plt.plot(x, w0, 'r-', label="w0")
    plt.ylabel("w0")
    plt.subplot(312)
    plt.plot(x, w1, 'g-', label="w1")
    plt.ylabel("w1")
    plt.subplot(313)
    plt.plot(x, w2, 'b-', label="w2")
    plt.ylabel("w2")
    plt.xlabel("迭代次数", FontProperties=font)
    plt.show()


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y.transpose())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plot_sigmoid():
    x = arange(-60.0, 60.0, 1)
    y = sigmoid(x)
    fig = plt.figure(figsize=(8, 4))
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.plot(x, y.transpose())
    plt.show()


if __name__ == "__main__":
    data_mat, label_mat = loadDataSet()
    weights, weights_list = gradAscent(data_mat, label_mat)
    #weights, weights_list = stocGradAscent(data_mat, label_mat)
    print(weights)
    #logistic_lib(data_mat, label_mat)
    #plot_weights(weights_list)
    #plotBestFit(weights)
    # plot_sigmoid()
