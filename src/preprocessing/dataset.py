import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.features import Feature


def load_data():
    fea = Feature
    evil = pd.read_csv("../data/xssed.csv")
    normal = pd.read_csv("../data/normal_examples.csv")
    data, label = fea.transform(evil, normal)

    standard_scaler = StandardScaler()

    # 样本标准化
    data = standard_scaler.fit_transform(data)

    return data, label