import numpy as np
from sklearn.svm import SVC
import joblib
import os
from sklearn.metrics import classification_report

from preprocess import XssData, Vec, ParseData


class Record:
    def __init__(self):
        self.word2vec = Vec()
        self.state = False

    def load_model(self):
        if not self.state:
            self.state = self.word2vec.load()
        return self.state

    def get_data(self, data):
        """
        :param data: np.array
        :return: np.array
        """
        embedding_size = self.word2vec.embedding_size
        xs = np.zeros([len(data), embedding_size])
        data = ParseData(data)

        if not self.load_model():
            self.word2vec.fit(data)
        for num, x in enumerate(data):
            xx = []
            for word in x:
                xx.append(self.word2vec.predict(word))

            xs[num] += np.array(xx).sum(axis=0)
        return xs


class Detect(Record):
    def __init__(self):
        super(Detect, self).__init__()
        self.svc = SVC() if not os.path.exists("model/svc-rbf.m") else joblib.load("model/svc-rbf.m")

    def fit(self, train):
        train_x, train_y = train[:, 0], train[:, -1]
        train_x = self.get_data(train_x)
        train_y = train_y.astype("float")
        self.svc.fit(train_x, train_y)
        joblib.dump(self.svc, "model/svc-rbf.m")

    def predict(self, param):
        """
        :param param: str, payload
        :return:
        """
        if isinstance(param, str):
            param = [param]
        param = np.array(param).ravel()
        x = self.get_data(param)
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
    data = ["%3c/title%3e%3cscript%3ealert(%22xss%22)%3c/script%3e", "COLLCC%3D3442798258%26"]
    print(data)
    y = detect.predict(data)
    print(y)


if __name__ == '__main__':
    detect = Detect()
    detect.predict()
    train_predict(detect)
    # test(detect)