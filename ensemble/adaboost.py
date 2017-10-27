import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import load_iris


def load_dataset(path):
    X = []
    y = []
    with open(path) as f:
        for line in f.readlines():
            line = line.split('\t')
            X.append(line[:-1])
            y.append(line[-1])
    return np.float64(X), np.float64(y)


class Adaboost():
    def __init__(self, algo, n_class=2):
        if n_class != 2:
            raise ValueError('adaboost只能二分类')
        self.__n_class = n_class
        self.__algo = algo
        self.__alpha = []

    def __sign(self, x):
        x = np.where(x > 0, 1, -1)
        x = np.where(x == 0, 0, x)
        return x

    def fit(self, X, y, iteration):
        num = len(X)
        if num != len(y):
            raise ValueError('样本跟标签不匹配')
        if iteration < 2:
            raise ValueError('迭代次数错误')

        weight = np.ones((num)) / num

        for i in range(iteration):
            self.__algo.fit(X, y)
            predict_y = self.__algo.predict(X)
            error_rate = np.sum(weight * (y != predict_y)) / num
            if error_rate > 0.5 or error_rate < 1e-7:
                break
            alpha = 0.5 * math.log((1 - error_rate) / error_rate)
            self.__alpha.append(alpha)
            temp = np.exp(-alpha * y * predict_y)
            weight = weight * temp / np.sum(temp)

    def predict(self, X):
        predict_y = self.__algo.predict(X)
        sum = np.zeros((len(X)))
        for alpha in self.__alpha:
            sum += alpha * predict_y
        return self.__sign(sum)


if __name__ == '__main__':
    train_X, train_y = load_dataset('data/horseColicTraining2.txt')
    test_X, test_y = load_dataset('data/horseColicTest2.txt')
    # train_X, train_y = load_iris(True)
    # train_X = pd.read_csv('data/train.csv', header=0,)
    # seq = np.random.permutation(train_y)[:100]
    # test_X, test_y = train_X[seq], train_y[seq]
    # # print(train_X.shape, train_y.shape)
    

    clf = DecisionTreeClassifier(random_state=1, max_depth=2)
    # clf = LogisticRegression(random_state=1)
    clf.fit(train_X, train_y)
    predict_y = clf.predict(test_X)
    print(np.sum(test_y == predict_y) / len(test_y))

    boost = Adaboost(DecisionTreeClassifier(random_state=1, max_depth=2))
    # boost = Adaboost(LogisticRegression(random_state=1))
    boost.fit(train_X, train_y, 100)
    predict_y = boost.predict(test_X)
    print(np.sum(test_y == predict_y) / len(test_y))
