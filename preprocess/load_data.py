import numpy as np
import pandas as pd
import joblib
from collections import Counter
import os

from sklearn.model_selection import StratifiedShuffleSplit
from preprocess.parser import GeneSeg

FEATURE = "cache/xss.feature"


def shuffle_split(data):
    """
    划分数据
    :param data: np.array两列数据
    :return: 数据索引
    """
    sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train, test in sp.split(np.arange(len(data)), data[:, 1]):
        return data[train], data[test]


def get_feature(data: np.array = np.empty([0]), num_feature=300):
    """
    获取特征，如果data存在数据，则将该数据映射为特征，不存在说明文件存在直接取文件读取
    :param data: list param没有解析过的数据
    :param num_feature: int 特征数目
    :return: dict 字典
    """
    if len(data) == 0:
        assert os.path.exists(FEATURE)
        return joblib.load(FEATURE)
    data = data[data[:, -1] == 1, 0]
    ans = []
    for x in data:
        ans.extend(GeneSeg(x))
    fea = Counter(ans).most_common(num_feature)
    fea, _ = zip(*fea)
    feature = {"WORD": 0}
    for elem in fea:
        feature[elem] = len(feature)
    joblib.dump(feature, FEATURE)
    return feature


class ParseData:
    feature = get_feature()

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            x = [elem if elem in self.feature.keys() else "WORD" for elem in GeneSeg(x)]
            yield x


class XssData:
    def __init__(self):
        self.normal = "data/normal_examples.csv"
        self.test = "cache/test"
        self.xssed = "data/xssed.csv"
        self.train = "cache/train"

        self.train, self.test = self.reader()
        get_feature(self.train)

    def reader(self):
        if os.path.exists(self.train) and os.path.exists(self.test):  # 存在划分的数据集直接加载
            train = joblib.load(self.train)
            test = joblib.load(self.test)
            return train, test

        normal = pd.read_csv(self.normal)
        xss = pd.read_csv(self.xssed)
        normal["label"] = 0.
        xss["label"] = 1.

        data = np.concatenate([normal, xss], axis=0)

        train, test = shuffle_split(data)

        train, test = np.array(data[train]), np.array(data[test])  # train， test为索引，将对应索引的文件划分
        joblib.dump(train, self.train)  # 存到文件中去
        joblib.dump(test, self.test)
        return train, test
