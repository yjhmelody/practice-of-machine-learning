import math

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier


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
    def __init__(self, algo, n_class=2, **kwargs):
        if n_class != 2:
            raise ValueError('adaboost只能二分类')
        self.__n_class = n_class
        self.__algo = algo(**kwargs)
        self.__alpha = []

    def __sign(self, x):
        x = np.copy(x)
        np.where(x > 0, 1, -1)
        np.where(x == 0, 0, x)
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
        # alpha = np.array(self.__alpha)
        predit_y = self.__algo.predict(X)
        print('sum', np.sum(predit_y))
        sum = np.zeros((len(X)))
        for alpha in self.__alpha:
            sum += alpha * predit_y
            print(sum)
        return self.__sign(sum)


if __name__ == '__main__':
    train_X, train_y = load_dataset('data/horseColicTraining2.txt')
    test_X, test_y = load_dataset('data/horseColicTest2.txt')

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_X, train_y)
    predit_y = clf.predict(test_X)
    print(np.sum(test_y == predit_y) / len(test_y))
    print('sum', np.sum(predit_y))

    boost = Adaboost(DecisionTreeClassifier, random_state=0)
    boost.fit(train_X, train_y, 0)
    predit_y = boost.predict(test_X)
    print(np.sum(test_y != predit_y) / len(test_y))
