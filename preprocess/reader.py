import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

from preprocess.parser import GeneSeg


NORMAL = "data/normal_examples.csv"
XSSED = "data/xssed.csv"
FEATURE = "cache/xss.feature"
TRAIN = "cache/train"
TEST = "cache/test"


def shuffle_split(x, y):
    """
    划分数据
    :param x:
    :param y: 类别标签
    :return: 数据索引
    """
    sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train, test in sp.split(np.arange(len(x)), y):
        return train, test


def get_feature(data=[], num_feature=300):
    """
    获取特征，如果data存在数据，则将该数据映射为特征，不存在说明文件存在直接取文件读取
    :param data: list param没有解析过的数据
    :param num_feature: int 特征数目
    :return: dict 字典
    """
    if len(data) == 0:
        return joblib.load(FEATURE)
    ans = []
    for x in data:
        ans.extend(GeneSeg(x))
    fea = Counter(ans).most_common(num_feature)
    fea, _ = zip(*fea)
    feature = {"WORD": 1}
    for elem in fea:
        feature[elem] = len(feature) + 1
    joblib.dump(feature, FEATURE)
    return feature


def reader():
    if os.path.exists(TRAIN) and os.path.exists(TEST):  # 存在划分的数据集直接加载
        train = joblib.load(TRAIN)
        test = joblib.load(TEST)
        return train, test

    normal = pd.read_csv(NORMAL)
    xssed = pd.read_csv(XSSED)
    normal["label"] = 0.
    xssed["label"] = 1.

    data = np.concatenate([normal, xssed], axis=0)

    train, test = shuffle_split(data, data[:, -1])
    train_data = data[train]
    train_data = train_data[train_data[:, -1] == 1, 0]
    get_feature(train_data)     # 使用训练集的数据获取特征，部分异常数据并不包含在内，以后数据将映射到次特征控件

    train, test = np.array(data[train]), np.array(data[test])  # train， test为索引，将对应索引的文件划分
    joblib.dump(train, TRAIN)   # 存到文件中去
    joblib.dump(test, TEST)
    return train, test
