#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np


class FullConnectedLayer():
    def __init__(self, input_size, output_size, activator):
        '''
        构造函数
        input_size: 本层输入向量维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 偏执项
        self.b = np.zeros(output_size)
        # 输出向量
        # self.output = np.zeros(output_size)

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度为input_size
        '''
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传入的误差项
        '''
        # self.delta是偏差项,用来反向计算self.delta
        self.delta = self.activator.backward(
            self.input) * np.dot(self.W.T, delta_array)

        self.W_grad = np.dot(delta_array, self.input)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad


class Sigmoid():
    def forward(self, weight):
        return 1.0 / (1.0 + np.exp(-weight))

    def backward(self, output):
        return output * (1 - output)


class Network():
    def __init__(self, layers):
        '''
        layers: 设置层数大小
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(
                layers[i], layers[i + 1], Sigmoid()))

    def predict(self, sample):
        '''
        sample: 预测样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, dataset, labels, rate, epoch):
        '''
        训练数据
        dataset: 输入集
        labels: 样本标签
        rate: 学习率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(dataset)):
                self.train_one_sample(dataset[d], labels[d], rate)

    def train_one_sample(self, label, data, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        # 输出层计算
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output) * (label - self.layers[-1].output)
        # 隐藏层计算
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)
