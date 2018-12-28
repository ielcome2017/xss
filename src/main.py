from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

from src.preprocessing.dataset import load_data


def train(kerels, x, y):
    for kerel in kerels:
        clf = SVC(gamma="auto", kerel=kerel)
        clf.predict(x, y)
        pickle.dump(clf, open("../model/train_model.msvm_"+kerel, "wb"))

def test(kerels, x, y):
    for kerel in kerels:
        clf = pickle.load(open("../model/train_model.msvm_"+kerel, "rb"))
        pred = clf.predict(x)
        print(classification_report(y, pred))


if __name__ == "__main__":
    data, label = load_data()

    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2)

    kerels = ['rbf', 'linear']

    test(kerels, test_x, test_y)
