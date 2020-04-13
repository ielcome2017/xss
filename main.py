import numpy as np
from sklearn.svm import SVC
import joblib
import os
from sklearn.metrics import classification_report

from preprocess.vec import Vec
from preprocess.load_data import XssData, parse_data


def get_data(data, word2vec):
    """
    传入解析后的数据，每一行记录为[str, ...]，需要转化为向量[[0.1, 0.2...]...]
    :param data: _Parse 解析后的数据，训练测试集数据
    :param word2vec: Vec word2vec转化工具，Vec.predict将被用于转化
    :return: 数值型数据
    """
    embedding_size = word2vec.embedding_size
    xs = np.zeros([len(data), embedding_size])

    data = parse_data(data)
    for num, x in enumerate(data):
        xx = []
        for word in x:
            xx.append(word2vec.predict(word))
        xs[num] += np.array(xx).sum(axis=0)
    return xs


class Detect:
    def __init__(self):
        self.svc = SVC() if not os.path.exists("model/svc-rbf.m") else joblib.load("model/svc-rbf.m")
        self.feature = joblib.load("cache/xss.feature")
        self.word2vec = Vec()
        if not self.word2vec.load():
            self.word2vec.fit()

    def fit(self, train):
        train_x, train_y = train[:, 0], train[:, -1]
        train_x = get_data(train_x, self.word2vec)
        train_y = train_y.astype("float")
        self.svc.fit(train_x, train_y)
        joblib.dump(self.svc, "model/svc-rbf.m")

    def predict(self, param):
        if isinstance(param, str):
            param = np.array([param])
        param = param.ravel()
        x = get_data(param, self.word2vec)
        y = self.svc.predict(x)
        return y


def train_predict(detect):
    xss_data = XssData()
    train, test = xss_data.train, xss_data.test
    detect.fit(train)

    test_x, test_y = test[:, 0], test[:, 1]
    test_y = test_y.astype("float")
    pred_y = detect.predict(test_x)
    print(classification_report(test_y, pred_y))


def test(detect):
    import pandas as pd
    # data = pd.read_csv("data/xssed.csv")
    # data = data.head(5).values
    data = "%3c/title%3e%3cscript%3ealert(%22xss%22)%3c/script%3e"
    print(data)
    y = detect.predict(data)
    print(y)


if __name__ == '__main__':
    detect = Detect()
    # train_predict(detect)
    test(detect)
