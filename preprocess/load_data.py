from preprocess.reader import reader, shuffle_split, get_feature
from preprocess.parser import GeneSeg


def transform(x, feature):
    """
    :param x: str 原始数据中param列下的数据，需要用GeneSeg解析，计息完后是各列表
    :param feature: dict
    :return: list 包含各字符的列表
    """
    x = GeneSeg(x)
    x = [elem if elem in feature.keys() else "WORD" for elem in x]
    return x


def split(data):
    """
    将数据划分为训练集和验证集
    :param data:_Parse 解析完后的数据
    :return:
    """
    train, val = shuffle_split(data, data[:, -1])
    return data[train], data[val]


class ParseX:
    def __init__(self, x):
        self.x = x
        self.feature = get_feature()

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        for x in self.x:
            x = transform(x, self.feature)
            yield x


class Record:
    def __init__(self, data, word2vec):
        self.data = data
        self.word2vec = word2vec

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            _x = []
            for word in x:
                _x.append(self.word2vec.predict(word))
            yield _x


class XssData:
    def __init__(self):
        """
        先加载数据，再将数据用_Parse解析, 解析后的训练集数据将被拿去训练word2vec
        """
        self.train, self.test = reader()


