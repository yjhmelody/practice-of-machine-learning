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


def gradient_ascent(X, y, alpha=0.00128, iterations=3000):
    '''梯度上升算法'''
    # 预处理
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64).transpose()

    X_T = X.T
    # theta初始化为长度为特征数的列向量
    theta = np.ones(X.shape[1])
    # 样本数
    m = X.shape[0]
    for iteration in range(iterations):
        h = sigmoid(np.dot(X, theta))
        loss = y - h
        # theta = theta + alpha * np.dot(X_T, loss) / m
        theta = theta + alpha * np.dot(X_T, loss)

    cost = np.sum(loss ** 2) / (2 * m)
    print('the cost is: ', cost)
    print('the theta is: ', theta)
    return theta


def SGA(X, y, init_alpha=0.01, iterations=300):
    '''随机梯度上升算法'''
    # 预处理
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64).transpose()

    m, n = X.shape
    theta = np.ones(n)
    # 每次随机选择一个来进行梯度上升
    # 总迭代次数是 iterations * m
    for iteration in range(iterations):
        for i in range(m):
            # 随迭代递减学习率
            alpha = 4 / (1 + i + iteration) + init_alpha
            # 从m个样本里随机选择一个
            rand_index = int(np.random.uniform(0, m))
            h = sigmoid(np.sum(X[rand_index] * theta))
            loss = y[rand_index] - h
            theta = theta + alpha * np.dot(X[rand_index], loss)

    return theta


def plot_best_fit(weights):
    import matplotlib.pyplot as plt

    data, label = load_dataset()
    data = np.array(data)

    n = np.shape(data)[0]
    x1, y1, x2, y2 = [], [], [], []

    for i in range(n):
        # str -> int
        if int(label[i]) == 1:
            x1.append(data[i, 1])
            y1.append(data[i, 2])
        else:
            x2.append(data[i, 1])
            y2.append(data[i, 2])
    plt.scatter(x1, y1, s=20, c='red')
    plt.scatter(x2, y2, s=20, c='green')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    plt.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    data, label = load_dataset()
    weights1 = gradient_ascent(data, label, alpha=0.0128, iterations=30000)
    print(weights1)
    weights2 = SGA(data, label, init_alpha=0.01, iterations=3000)
    print(weights2)
    plot_best_fit(weights1)
    plot_best_fit(weights2)
