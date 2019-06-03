import pandas as pd
from geneseg import GeneSeg
import numpy as np
import pickle
from gensim.models.word2vec import Word2Vec
import os


class Feature(object):

    def transform(self, evil, normal):
        """
        :param evil: pd.DateFrame
        :param normal: pd.DateFrame
        :return:
        """
        evil, normal = self.generate_data_word(evil, normal)

        n_samples = 12039

        evil_data = self.word2vec(evil, n_samples)
        normal_data = self.word2vec(normal, n_samples)

        data = pd.concat([evil_data, normal_data], axis=0)
        label = np.r_[np.ones(n_samples), np.zeros(n_samples)]

        del evil, normal, evil_data, normal_data

        return data, label

    def word2vec(self, evil, n_samples, model_path='model/preprocessing/model_word2Vec_auto'):
        embedding_size = 128
        skip_windows = 5
        num_sampled = 64
        num_iter = 100

        # 加载或者训练存储模型
        if os.path.exists(model_path):
            model = Word2Vec.load(model_path)
        else:
            model = Word2Vec(evil, size=embedding_size, window=skip_windows, negative=num_sampled, iter=num_iter)
            model.save(model_path)
        embeddings = model.wv
        dictionary = dict([(embeddings.index2word[i], i) for i in range(len(embeddings.index2word))])

        # reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
        # word2vec={"dictionary":dictionary,"embeddings":embeddings,"reverse_dictionary":reverse_dictionary}
        def generate_vec(words):
            l = np.zeros(embedding_size)
            for word in words:
                if word in dictionary:
                    l += np.array(embeddings[word])
            return l

        # word向量化
        black = evil.apply(lambda x: generate_vec(x))

        black_percent10 = black.sample(n=n_samples)

        del black

        return black_percent10

    def generate_data_word(self, evil: pd.DataFrame, normal: pd.DataFrame):
        """
        把一整条语句分成一个一个的词
        :param evil: 这里标注evil是表示第一次使用evil训练模型，后面不管是正常的还是攻击记录都会使用该模型
        :return: 10%的数据
        """

        get_evil_word = self.get_evil_word      #获取特征

        # 用正则表达式提取语句中的参数
        normal_parsed = normal['param'].map(GeneSeg)
        evil_parsed = evil['param'].map(GeneSeg)

        # 获取特征空间
        evil_word = get_evil_word(evil_parsed)

        # 将不在特征空间的词全部泛化为WORD
        evil_data = evil_parsed.apply(lambda x: [i if i in evil_word else 'WORD' for i in x])
        normal_data = normal_parsed.apply(lambda x: [i if i in evil_word else 'WORD' for i in x])
        return evil_data, normal_data

    def get_evil_word(self, evil_parsed, model_path="model/preprocessing/evil_word"):
        # 提取evil['parsed']中的词集，存到evil_word.csv文件中
        if os.path.exists(model_path):
            features = pickle.load(open(model_path, "rb"))
        else:
            ans = []
            evil_parsed.apply(lambda x: [ans.append(i) for i in x])

            evil_word = pd.value_counts(ans)[:300]
            features = np.array(evil_word.index)

            pickle.dump(features, open(model_path, "wb"))

        return features


if __name__ == "__main__":
    fea = Feature()
    evil = pd.read_csv("data/xssed.csv")
    normal = pd.read_csv("data/normal_examples.csv")
    data, label = fea.transform(evil, normal)

    print(data.iloc[0])