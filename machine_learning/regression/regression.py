# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-11-24 16:32:31
# @Last Modified by:   huzhu
# @Last Modified time: 2019-11-25 21:06:40

import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    num_feat = len(open(file_path).readline().split("\t")) - 1
    data_mat = list()
    lable_mat = list()
    fr = open(file_path)
    for line in fr.readlines():
        line_arr = list()
        cur_line = line.strip().split("\t")
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        lable_mat.append(float(cur_line[-1]))
    return data_mat, lable_mat


def stand_regres(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    xTx = x_mat.T * x_mat
    ws = np.linalg.solve(xTx, x_mat.T * y_mat)
    return ws


def lwlr(test_point, x_arr, y_arr, k=1.0):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    n = np.shape(x_mat)[0]
    weights = np.mat(np.eye(n))
    for i in range(n):
        diff_mat = test_point - x_mat[i, :]
        weights[i, i] = np.exp((diff_mat * diff_mat.T) / (-2 * k ** 2))
    xTx = x_mat.T * weights * x_mat
    ws = np.linalg.solve(xTx, x_mat.T * weights * y_mat)
    return test_point * ws


def lwrl_test(test_arr, x_arr, y_arr, k=1.0):
    m = np.shape(test_arr)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat

def ridge_regres(x_mat, y_mat, lam=0.2):
    xTx = x_mat.T * x_mat
    m = np.shape(x_mat)[1]
    denom = xTx + np.eye(m) * lam
    ws = np.linalg.solve(denom, x_mat.T * y_mat)
    return ws

def ridge_test(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    y_mean = np.mean(y_mat, 0)
    y_mat -= y_mean
    x_mean = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_mean) / x_var
    num_test_pts = 30
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regres(x_mat, y_mat, np.exp(i -10))
        w_mat[i, :] = ws.T
    print(w_mat)
    return w_mat 

def regularize(xMat):
    inMat = xMat.copy()
    #calc mean then subtract it off
    inMeans = np.mean(inMat,0) 
    #calc variance of Xi then divide by it  
    inVar = np.var(inMat,0)      
    inMat = (inMat - inMeans)/inVar
    return inMat

def rss_error(yArr,yHatArr): 
    #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def stage_wise(x_arr, y_arr, eps = 0.01, num_it = 100):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = regularize(x_mat)
    m,n = np.shape(x_mat)
    return_mat= np.zeros((num_it, n))
    ws = np.zeros((n, 1))
    ws_test = ws.copy()
    ws_mat = ws.copy()
    for i in range(num_it):
        print(ws.T)
        lse = np.inf
        for j in range(n):
            for sign in [-1,1]:
                ws_test = ws.copy()
                ws_test[j] += sign * eps
                y_test = x_mat * ws_test
                rsse = rss_error(y_mat.A, y_test.A)
                if rsse < lse:
                    lse = rsse 
                    ws_mat = ws_test
        ws = ws_mat.copy()
        return_mat[i,:] = ws.T
    return return_mat

def test01():
    x_arr, y_arr = load_data("ex0.txt")
    ws = stand_regres(x_arr, y_arr)
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    y_hat = x_mat * ws
    plt.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])
    xcopy = x_mat.copy()
    xcopy.sort(0)
    y_hat = xcopy * ws
    plt.plot(xcopy[:, 1], y_hat)
    y_hat = lwrl_test(x_arr, x_arr, y_arr, k=0.01)
    srt_ind=x_mat[:, 1].argsort(0)
    print(y_hat[srt_ind])
    plt.plot(xcopy[:,1],y_hat[srt_ind], c = "red")
    plt.show()

def test02():
    ab_x, ab_y = load_data("abalone.txt")
    ridge_weights = ridge_test(ab_x, ab_y)
    plt.plot(ridge_weights)
    plt.show()

def test03():
    ab_x, ab_y = load_data("abalone.txt")
    stage_wise(ab_x, ab_y, 0.01, 200)

if __name__ == '__main__':
    #test01()
    test02()
    test03()
#