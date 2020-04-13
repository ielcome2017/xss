from preprocess.reader import reader, shuffle_split, get_feature
from preprocess.parser import GeneSeg


def split(data):
    """
    将数据划分为训练集和验证集
    :param data:_Parse 解析完后的数据
    :return:
    """
    train, val = shuffle_split(data, data[:, -1])
    return data[train], data[val]


class ParseData:
    def __init__(self, data, feature=get_feature()):
        self.data = data
        self.feature = feature

    def __iter__(self):
        for x in self.data:
            x = [elem if elem in self.feature.keys() else "WORD" for elem in GeneSeg(x)]
            yield x


class XssData:
    def __init__(self):
        """
        先加载数据，再将数据用_Parse解析, 解析后的训练集数据将被拿去训练word2vec
        """
        self.train, self.test = reader()


