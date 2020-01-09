# -*- coding: utf-8 -*-
'''
@Author: huzhu
@Date: 2020-01-07 19:43:08
@Description: 简单的神经网络实现
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


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def one_hot(label_arr, n_samples, n_classes):
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), label_arr.T] = 1
    return one_hot


def plot_loss(loss):
    fig = plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(all_loss)), all_loss)
    plt.title("Development of loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()

class NeuralNetWork():
    def __init__(self, epsilon = 1, iters = 1000, alpha = 0.01, lam = 0.01):
        self.hidden_dim = 10    # 默认取 4 个隐藏层神经元
        self.n_hidden = 1       # 默认取 1 个隐藏层
        self.epsilon = epsilon  # 随机初始化权值矩阵的界限
        self.iters = iters      # 最大迭代次数
        self.alpha = alpha      # 学习率
        self.lam = lam          # 正则化项系数
        self.weights = list()   # 初始化权值矩阵
        self.gradients = list() # 初始化梯度矩阵
        self.bias = 1           # 偏置项


    def init_weights(self, n_input, n_output):
        # 第 1 → 2 层的权值矩阵
        self.weights.append((2 * self.epsilon) * np.random.rand(self.hidden_dim, n_input + 1) - self.epsilon)
        # 第 2 → n 层的权值矩阵
        for i in range(self.n_hidden - 1):
            self.weights.append((2 * self.epsilon) * np.random.rand(self.hidden_dim, self.hidden_dim + 1) - self.epsilon)
        self.weights.append((2 * self.epsilon) * np.random.rand(n_output, self.hidden_dim + 1) - self.epsilon)
    

    def init_gradients(self, n_input, n_output):
        # 第 1 → 2 层的梯度矩阵
        self.gradients.append(np.zeros((self.hidden_dim, n_input + 1)))
        # 第 2 → n 层的梯度矩阵
        for i in range(self.n_hidden - 1):
            self.gradients.append(np.zeros((self.hidden_dim, self.hidden_dim + 1)))
        self.gradients.append(np.zeros((n_output, self.hidden_dim + 1)))


    def train(self, data_arr, label_arr):
        n_samples, n_features = data_arr.shape
        n_output = len(set(label_arr.flatten()))   # 输出类别个数
        y_one_hot = one_hot(label_arr, n_samples, n_output)
        all_loss = list()  # 损失函数记录
        self.init_weights(n_features, n_output)
        self.init_gradients(n_features, n_output)
        for it in range(self.iters):
            for index in range(n_samples):
                # 计算前向传播每一层的输出值
                layer_output = self.forward_propagation(data_arr[index])
                # 计算每一层的误差
                layer_error = self.cal_layer_error(layer_output, y_one_hot[index])
                # 计算梯度矩阵
                self.cal_gradients(layer_output, layer_error)
            # 更新权值
            self.update_weights(n_samples)
            # 累加输出误差
            #loss = self.cal_loss(data_arr, y_one_hot, n_samples)
            #all_loss.append(loss)
        return self.weights, all_loss

    
    def forward_propagation(self, data):
        layer_output = list()
        a = np.insert(data, 0, self.bias)
        layer_output.append(a)
        for i in range(self.n_hidden + 1):
            z = self.weights[i] @ a
            a = sigmoid(z)
            if i != self.n_hidden:
                a = np.insert(a, 0, self.bias)
            layer_output.append(a)
        return np.array(layer_output)
    

    def cal_layer_error(self, layer_output, y):
        # 只有第 2 →n 层有误差，输入层没有误差
        layer_error = list()
        # 计算输出层的误差
        error = layer_output[-1] - y
        layer_error.append(error)
        # 反向传播计算误差
        for i in range(self.n_hidden, 0, -1):
            error = self.weights[i].T @ error * layer_output[i] * (1 - layer_output[i])
            # 删除第一项，偏置项没有误差
            error = np.delete(error, 0)
            layer_error.append(error)
        return np.array(layer_error[::-1])
    

    def cal_gradients(self, layer_output, layer_error):
        for l in range(self.n_hidden + 1):
            for i in range(self.gradients[l].shape[0]):
                for j in range(self.gradients[l].shape[1]):
                    self.gradients[l][i][j] += layer_error[l][i] * layer_output[l][j]


    def update_weights(self, n_samples):
        for l in range(self.n_hidden + 1):
            gradient = 1.0 / n_samples * self.gradients[l] + self.lam * self.weights[l]
            gradient[:,0] -= self.lam * self.weights[l][:,0]
            self.weights[l] -= self.alpha * gradient


    def cal_loss(self, data_arr, y_one_hot, n_samples):
        loss = 0  # 这里不用添加正则化项
        for i in range(n_samples):
            y = y_one_hot[i]
            output = self.forward_propagation(data_arr[i])[-1]
            loss += np.sum((y * np.log(output) + (1 - y) * np.log(1 - output)))
        loss = (-1 / n_samples) * loss
        return loss
    

    def predict(self, data_arr):
        n_samples = data_arr.shape[0]
        ret = np.zeros(n_samples)
        for i in range(n_samples):
            output = self.forward_propagation(data_arr[i])[-1]
            ret[i] = np.argmax(output)
        return ret.reshape((-1,1))
    

    def plot_decision_boundary(self, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(y))
        plt.show()


if __name__ == "__main__":
    # 加载数据
    train_data_arr, train_label_arr = load_dataset('train_dataset.txt')
    test_data_arr, test_label_arr = load_dataset('test_dataset.txt')

    # 训练数据
    nn = NeuralNetWork(iters = 1000)
    weights, all_loss = nn.train(train_data_arr, train_label_arr)
    #print(weights)
    y_predict = nn.predict(test_data_arr)
    accurcy = np.sum(y_predict == test_label_arr) / len(test_data_arr)
    print(accurcy)

    # 绘制决策边界
    #nn.plot_decision_boundary(test_data_arr, test_label_arr)
    # 绘制损失函数
    #plot_loss(all_loss)
