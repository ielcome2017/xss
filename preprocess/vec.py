import os
from gensim.models.word2vec import Word2Vec
import joblib
import sys
import numpy as np

WORD2VEC = "cache/xss.word2vec"
FILENAME = "cache/xss.feature"


def func(x, feature):
    return feature.get(x)


class Word:
    def __init__(self, data):
        self.data = data
        self.epoch = 0

    def __iter__(self):
        count = 0
        for x, y in self.data:
            sys.stdout.write("\rEpoch: {}, STEP: {}/{}".format(self.epoch, count, len(self.data)))
            sys.stdout.flush()
            count += 1
            yield x
        self.epoch += 1


class Vec:
    def __init__(self, embedding_size=128, skip_windows=5, num_sampled=64, num_iter=100):
        self.embedding_size = embedding_size
        self.skip_windows = skip_windows
        self.num_sampled = num_sampled
        self.num_iter = num_iter
        self.model = None
        self.embedding = None

    def fit(self, word):
        if self.load():
            return self.model
        word = Word(word)
        self.model = Word2Vec(
            word,
            size=self.embedding_size,
            window=self.skip_windows,
            negative=self.num_sampled,
            iter=self.num_iter
        )
        self.model.save(WORD2VEC)
        self.embedding = self.model.wv
        return self.model

    def predict(self, x):
        if x in self.model.wv.index2word:
            return self.embedding[x]
        return np.zeros(self.embedding_size)

    def load(self, filename=WORD2VEC):
        if os.path.exists(filename):
            self.model = Word2Vec.load(WORD2VEC)
            self.embedding = self.model.wv
            return True
        print("current file is not exist!")
        return False
