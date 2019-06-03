#%%
import nltk
import re
from urllib import parse
#%%
def GeneSeg(payload):
    #数字泛化为"0"
    payload=payload.lower()
    payload=parse.unquote(parse.unquote(payload))
    payload,num=re.subn(r'\d+',"0",payload)
    #替换url为”http://u
    payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)

#%%
import pandas as pd 
evil = pd.read_csv("data/xssed.csv")
evil_parsed = evil["param"].map(GeneSeg)
print(evil_parsed.head(5))

#%%
import numpy as np 
def get_evil_word(evil_parsed):
    ans = []
    evil_parsed.apply(lambda x: [ans.append(i) for i in x])
    evil_word = pd.value_counts(ans)[: 300]
    print(evil_word)
    features = np.array(evil_word.index)
    return features

#%%
evil_word = get_evil_word(evil_parsed)
print(evil_word)

#%%
evil_data = evil_parsed.apply(lambda x: [i if i in evil_word else "WORD" for i in x])

#%%
from gensim.models.word2vec import Word2Vec

#%%
model = Word2Vec(evil_data, window=5, negative=64, iter=100)
embeddings = model.wv

#%%
max_length = max(len(i) for i in evil_parsed)
    

#%%
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, data, label, embeddings):
        self._data = data
        self._label = label
        self._dictionary = dict([(embeddings.index2word[i], i) for i in range(len(embeddings.index2word))])
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        def generate_vec(words):
            l = np.zeros([len(words), 100], dtype=np.float32)
            for i, word in enumerate(words) :
                if word in self._dictionary:
                    l[i] = np.array(embeddings[word])
            return l
        x = generate_vec(self._data[index]) 
        y = self._label
        ori_length = x.shape[0]
        n_dim = x.shape[1]
        if ori_length < max_length:
            npi = np.zeros(n_dim, dtype=np.float32)
            npi = np.tile(npi, (max_length - ori_length,1))
            x = np.row_stack((x, npi))

        return x, y, ori_length



#%%
ds = Data(evil_data, 1, embeddings)
#%%
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset=ds, 
    batch_size=2, 
    pin_memory=True, 
    shuffle=True,
    drop_last=True,
)

#%%
for x, y, l in loader:
    print(np.shape(x))
    print(np.shape(y))
    print(np.shape(l))
    break

#%%
import torch.nn as nn
import torch
_, idx_sort = torch.sort(l, dim=0, descending=True)
x = x.index_select(0, idx_sort)
lengths = list(l[idx_sort])
pack = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
print(pack)

#%%
