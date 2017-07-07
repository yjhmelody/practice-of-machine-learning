'''
1.收集数据
2.准备数据，只适用于标称型
3.分析数据，构造树完成后，应该检查图形是否符合预期
4.训练算法，构造树的数据结构
5.测试算法，使用经验树计算错误率
6.使用算法，决策树可以更好地理解数据内在含义
'''

import math
import operator
import pickle

import matplotlib.pyplot as plt


def create_dataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def entropy(dataset):
    '''计算数据集的信息熵'''
    label_counts = {}
    for vector in dataset:
        # 每个向量最后一个值存放标签
        label = vector[-1]
        # 计算各种类别的个数
        if label not in label_counts.keys():
            label_counts[label] = 1
        else:
            label_counts[label] += 1
    # 信息熵
    ent = 0
    num = len(dataset)
    for key in label_counts:
        # 分类的概率
        p = label_counts[key] / num
        ent -= p * math.log(p, 2)
    return ent


def split_dataset(dataset, axis, value):
    '''按照给定特征(axis对应的特征是否等于value)划分数据集'''

    ret_dataset = []
    for feature_vector in dataset:
        # axis表示划分的位置
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            ret_dataset.append(reduced_feature_vector)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    '''选择最好特征划分，返回该特征的下标'''
    feature_num = len(dataset[0]) - 1
    base_entropy = entropy(dataset)
    best_info_gain = 0
    best_feature = -1
    for i in range(feature_num):
        # 数据集某种特征的所有值
        feature = [example[i] for example in dataset]
        # 创建唯一的分类标签集合
        unique_feature = set(feature)
        new_entropy = 0
        # 计算每种划分方式的信息熵
        for value in unique_feature:
            sub_dataset = split_dataset(dataset, i, value)
            # 计算该特征某个值在该特征的比例
            p = len(sub_dataset) / len(dataset)
            new_entropy += p * entropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        # 计算最好的信息增益
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(class_list):
    '''多数表决决定分类，返回该类别'''
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 1
        else:
            class_count[vote] += 1
        # print(class_count[vote], vote)

    sorted_class_count = sorted(
        class_count.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_class_count)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    '''递归创建决策树'''
    # 每个数据对应的分类
    class_list = [example[-1] for example in dataset]
    # 类别完全相同则停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        # 返回当前类别
        return class_list[0]
    # 划分到只有一个特征时
    if len(dataset[0]) == 1:
        # 返回多数表决的类别
        return majority_count(class_list)
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    tree = {
        best_feature_label: {}
    }
    new_labels = labels[:]
    # 用该特征分类后删除该特征
    del(new_labels[best_feature])
    # 获取该样本最好特征存在的类别
    unique_feature_values = set(example[best_feature] for example in dataset)

    for value in unique_feature_values:
        sub_labels = new_labels[:]
        tree[best_feature_label][value] = create_tree(
            split_dataset(dataset, best_feature, value), sub_labels)

    return tree

def classify(input_tree, feature_label, test_datset):
    '''用决策树进行分类'''
    # 第一个分类特征 即前序遍历
    for key in input_tree.keys():
        first_feature = key
        break
    # 第一次分类后的字典
    sencond_dict = input_tree[first_feature]
    # 找到在第一个分类在特征标签里的下标
    first_feature_index = feature_label.index(first_feature)
    for key in sencond_dict.keys():
        if test_datset[first_feature_index] == key:
            # 判断该键存储的值是否是字典
            if type(sencond_dict[key]).__name__ == 'dict':
                class_label = classify(sencond_dict[key], feature_label, test_datset)
            else:
                class_label = sencond_dict[key]
    return class_label


decision_node = {
    'boxstyle': 'sawtooth',
    'fc': '0.8'
}

leaf_node = {
    'boxstyle': 'round4',
    'fc': '0.8'
}

arrow_args = {
    'arrowstyle': '<-'
}


def plot_node(node_text, center_point, parent_point, node_type):
    create_plot.ax1.annotate(node_text, xy=parent_point, xytext=center_point, 
    textcoords='axes fraction', va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


def create_plot():
    figure = plt.figure(1, facecolor='white')
    figure.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node(u'决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node(u'决策节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


# 利用 pickle 模块持久化训练好的决策树
def store_tree(input_tree, filename):
    with open(filename, 'w') as f:
        pickle.dump(str(input_tree), f)


def recover_tree(filename):
    with open(filename, 'r') as f:
        return pickle.load(f)


def test_lenses():
    with open('lenses.txt') as f:
        lenses = [line.strip().split('\t') for line in f.readlines()]
        lenses_label = ['age', 'prescript', 'astigmatic', 'tearRate']
        lenses_tree = create_tree(lenses, lenses_label)
        print('lenses tree: ', lenses_tree)

if __name__ == '__main__':
    dataset, labels = create_dataset()
    tree = create_tree(dataset, labels)
    print(tree)
    print(dataset)
    print(labels)

    print(classify(tree, labels, [1, 0]))
    print(classify(tree, labels, [1, 1]))

    # create_plot()

    # store_tree(tree, 'classify_tree.txt')
    # print(recover_tree('classify_tree.txt'))

    test_lenses()

