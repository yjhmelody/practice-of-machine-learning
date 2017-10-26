'''
1.收集数据，使用任意方法
2.准备数据，需要数值型来计算距离，也可以讲标称数据转换为二值型数据计算距离
3.分析数据，使用任意方法
4.训练数据，无监督学习没有训练过程
5.测试算法，应用聚类算法，观察结果，可以使用量化误差指标来评价算法的结果
6.使用算法，通常情况下，簇质心可以代表整个簇的数据来做出决策
'''

import numpy as np


def load_dataset(filename):
    dataset = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            line = map(float, line)
            dataset.append(line)
    return dataset


def distance(A, B):
    '''返回A B的欧几里德距离'''
    return np.sqrt(np.sum(np.square(A - B)))


def rand_centers(dataset, k):
    '''随机生成k个簇心'''
    n = dataset.shape[1]
    min_feature = np.min(dataset, axis=0)
    max_feature = np.max(dataset, axis=0)
    print(min_feature.shape)
    centers = min_feature + (max_feature - min_feature) * np.random.rand(k, n)
    return centers


def kmeans(dataset, k=2, distance=distance, rand_centers=rand_centers):
    '''k-means算法'''
    pass
