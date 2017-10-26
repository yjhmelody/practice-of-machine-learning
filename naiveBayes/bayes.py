'''
1.收集数据
2.准备数据，需要数值型或者布尔型数据
3.分析数据，有大量特征时，绘制特征作用不大，此时使用直方图更好
4.训练算法，计算不同的独立特征的条件概率
5.测试算法，计算错误率
6.使用算法
'''

import numpy as np

def create_dataset():
    dataset = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1代表侮辱性文字， 0代表正常言论
    labels = [0, 1, 0, 1, 0, 1]
    return dataset, labels


def create_vocabulary_list(dataset):
    '''创建一个不重复的数据集list'''
    vocabulary = set()
    for document in dataset:
        # 创建数据集的并集
        vocabulary = vocabulary | set(document)
    return list(vocabulary)


def set_of_words_to_vector(vocabulary, document):
    '''输入词汇表跟文档，返回文档向量，表示词汇表的单词在文档中是否出现'''
    return_vector = [0] * len(vocabulary)
    for word in document:
        if word in vocabulary:
            return_vector[vocabulary.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return return_vector


# p(ci | w) = p(w | ci) * p(ci) / p(w)
def naive_bayes(dataset, labels):
    '''输入为文档矩阵，文档标签向量'''
    num_documents = len(dataset)
    num_words = len(dataset[0])
    p_abusive = sum(labels) / num_documents
    p0_num_words = np.zeros(num_words)
    p1_num_words = np.zeros(num_words)
    p0_denom = 0
    p1_denom = 0
    for i in range(num_documents):
        # 类别为1时
        if labels[i] == 1:
            p1_num_words += dataset[i]
            p1_denom += sum(dataset[i])            
        else:
            p0_num_words += dataset[i]
            p0_denom += sum(dataset[i])
    p1 = p1_num_words / p1_denom
    p0 = p0_num_words / p0_denom
    return p0, p1, p_abusive

if __name__ == '__main__':
    dataset, labels = create_dataset()
    vocabulary = create_vocabulary_list(dataset)
    # print(vocabulary)
    train_matrix = []
    for data in dataset:
        train_matrix.append(set_of_words_to_vector(vocabulary, data))
    p0, p1, p_abusive = naive_bayes(train_matrix, labels)
    print(p0)
    print(p1)
    print(p_abusive)