'''
1.收集数据
2.准备数据，格式转换
3.分析数据
4.训练数据，kNN不适用
5.测试数据
6.使用算法
'''

import operator
import os

import numpy as np
from matplotlib import pyplot as plt


def create_dataset():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify(input_X, dataset, labels, k):
    # input_X 测试数据
    # dataset 训练数据
    if k < 1:
        return None
    dataset_size = dataset.shape[0]
    # 用input_X来当中心点
    diff_matrix = np.tile(input_X, (dataset_size, 1)) - dataset
    # 欧式距离计算
    distances = (diff_matrix ** 2).sum(axis=1) ** 0.5
    # 排序为了选取距离最近的k个点
    sorted_distances_index = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distances_index[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # k近邻选好按照第二个元素的次序对元组进行降序排序
    sorted_class_count = sorted(
        class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回频率最高的元素标签
    return sorted_class_count[0][0]


def auto_normal(dataset):
    '''归一化特征值'''
    # newValue = (oldValue - minValue) / (maxValue - minValue)
    min_val = dataset.min(0)
    max_val = dataset.max(0)
    ranges = max_val - min_val
    normal_dataset = np.zeros(dataset.shape)
    shape = dataset.shape[0], 1
    normal_dataset = (dataset - np.tile(min_val, shape)) / \
        np.tile(ranges, shape)

    return normal_dataset, ranges, min_val


def file_to_matrix(filename):
    # 样本
    # 每年里程  游戏时间百分比  消费冰淇淋公升    喜欢程度
    # 40920     8.326976	0.953952	largeDoses
    # 14488	    7.153469	1.673904	smallDoses
    # 26052	    1.441871	0.805124	didntLike
    # 75136	    13.147394	0.428964	didntLike
    # 38344	    1.669788	0.134296	didntLike
    with open(filename) as f:
        lines = f.readlines()
        length = len(lines)
        # 数据是 length * 3
        matrix = np.zeros((length, 3))
        labels = []
        i = 0
        for line in lines:
            line = line.strip()
            # 3列数据用tab分隔
            data = line.split('\t')
            matrix[i, :] = data[0:3]
            labels.append(int(data[-1]))
            i += 1
        return matrix, labels


def plt_dating(x, y, labels):
    # 绘制约会数据散点图，并根据labels区分
    plt.scatter(x, y, 10 * np.array(labels), 10 * np.array(labels))
    plt.show()


def classify_person():
    '''约会喜欢程度预测函数'''
    # 类别
    result = ('not at all', 'in small doses', 'in large doses')
    # 读取训练数据
    path = './datingTestSet2.txt'
    dating_matrix, dating_labels = file_to_matrix(path)
    # 归一化特征值
    normal_matrix, ranges, min_val = auto_normal(dating_matrix)
    # 收集个人数据
    person_data = []
    person_data.append(input('frquent flier miles earned per year?'))
    person_data.append(input('percentage of time spend playing game?'))
    person_data.append(input('liters of ice-cream consumed per year?'))
    # person_data 需要归一化
    # 训练后的数据是浮点数float64，所以运算前需要先将输入数据转换为浮点数
    person_data = np.array(person_data, dtype=np.float64)
    # 分类后的标签
    classifier_result = classify(
        (person_data - min_val) / ranges, normal_matrix, dating_labels, 3)
    print('you will probably like this person: ',
          result[classifier_result - 1])


def test_classify_rate():
    path = './datingTestSet2.txt'
    dating_matrix, labels = file_to_matrix(path)
    # print(dating_matrix, labels)
    # plt_dating(dating_matrix[:, 0], dating_matrix[:, 1], labels)
    normal_matrix, ranges, min_val = auto_normal(dating_matrix)
    print(normal_matrix, ranges, min_val)
    ratio = 0.07
    m = normal_matrix.shape[0]
    # 测试数据量
    test_num = int(m * ratio)
    # 记录错误率
    err_count = 0
    for i in range(test_num):
        classifier_result = classify(
            normal_matrix[i, :], normal_matrix[test_num:m, :], labels[test_num:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' %
              (classifier_result, labels[i]))
        if classifier_result != labels[i]:
            err_count += 1
    print('the total error rate is: %f' % (err_count / test_num))


def img_to_vector(filename):
    # 32*32图片文本转换为1*1024向量
    vector = np.zeros((1, 1024))
    with open(filename) as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                vector[0, 32*i+j] = int(line[j])
    return vector

def handwriting_class_test():
    labels = []
    # 获取训练数据目录
    training_file_list = os.listdir('trainingDigits')
    # 训练数据量
    m = len(training_file_list)
    training_matrix = np.zeros((m, 1024))
    for i in range(m):
        filename = training_file_list[i]
        # 获取该文本对应的数字
        num_string = int(filename.split('_')[0])
        labels.append(num_string)
        # 添加数据到训练矩阵
        training_matrix[i, :] = img_to_vector('./trainingDigits/%s' % filename)
    # 测试数据
    test_file_list = os.listdir('testDigits')
    err_count = 0
    # 测试数据量
    test_m = len(test_file_list)
    for i in range(test_m):
        filename = test_file_list[i]
        # 测试数据对应的数字
        num_string = int(filename.split('_')[0])
        test_vector = img_to_vector('./testDigits/%s' % filename)
        classifier_result = classify(test_vector, training_matrix, labels, 3)
        # print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, num_string))
        if classifier_result != num_string:
            err_count += 1
            print(classifier_result,' != ', num_string)
    print('the total number of errors is: %d' % err_count)
    print('the total error rate is: %f' % (err_count / test_m))


if __name__ == '__main__':
    # group, labels = create_dataset()
    # label = classify([0, 0], group, labels, 3)
    # print(label)

    # test_classify_rate()

    # classify_person()

    handwriting_class_test()
