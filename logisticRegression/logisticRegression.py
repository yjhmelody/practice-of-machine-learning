#-*- coding:utf-8 -*-

'''
1.收集数据
2.准备数据，由于需要计算距离，因此要求数值型，另外，结构化数据最佳
3.分析数据，采用任意方法
4.训练算法，大部分时间用于训练，目的是为找到最佳的分类回归系数
5.测试算法，一旦训练完成，分类将会很快
6.使用算法，输入一些数据并转换为对应的结构化数据；
基于训练好的回归系数可以对这些数据进行简单的回归计算，并判定类别；
之后就可以在输出的类别上做其他分析工作
'''

import numpy as np

def load_dataset():
    data = []
    label = []
    with open('testSet.txt') as f:
        for line in f.readlines():
            line = line.strip().split()
            data.append([1, line[0], line[1]])
            label.append(line[2])
    return data, label

def sigmoid(z):
    '''类跃阶函数'''
    return 1 / (1 + np.exp(-z))

def gradient_ascent(X, y, alpha=0.0125, iterations=10000):
    '''梯度上升算法'''
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64).transpose()
    # theta初始化为长度为特征数的列向量
    theta = np.ones(X.shape[1])
    for iteration in range(iterations):
        h = sigmoid(np.dot(X, theta))
        # print('X', X.shape)
        # print('theta', theta.shape)
        # print('X * theta', np.dot(X, theta).shape)
        # print('y', y.shape)
        # print('h', h.shape)
        error = np.subtract(y, h)
        # print('error', error.shape)
        # print('X', X.T.shape)
        theta = theta + alpha * X.transpose().dot(error)
        print(theta)
    return theta

if __name__ =='__main__':
    data, label = load_dataset()
    weights = gradient_ascent(data, label)
    print(weights)
    