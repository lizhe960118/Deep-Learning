import io
import os
import math
import copy
import pickle
import zipfile
from textwrap import wrap
from pathlib import Path
from itertools import zip_longest
from collections import defaultdict
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

def read_data(path):
    files = {}
    for filename in path.glob('*'):
        if filename.suffix == '.csv':
            files[filename.stem] = pd.read_csv(filename)
        elif filename.suffix == '.dat':
            if filename.stem == 'ratings':
                columns = ['userId', 'movieId','rating', 'timestamp']
            else:
                columns = ['movieId', 'title','genres']
            data = pd.read_csv(filename, sep='::',names=columns,engine='python')
            files[filename.stem] = data
    return files['ratings'], files['movies']

download_path = Path.home() / 'project'/ 'movie'/ 'data' / 'movielens'

ratings, movies = read_data(download_path / 'ml-1m')

def create_dataset(ratings, top=None):
    if top is not None:
        ratings.groupby('userId')['rating'].count()
    
    unique_users = ratings.userId.unique()
    user_to_index = {old:new for new, old in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)
    
    unique_movies = ratings.movieId.unique()
    movie_to_index = {old:new for new,old in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)
    
    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]
    
    X = pd.DataFrame({'user_id':new_users, 'movie_id':new_movies})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_movies), (X, y),(user_to_index, movie_to_index)

(n,m),(X, y),_ = create_dataset(ratings)

class ReviewsIterator:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        X, y = np.asarray(X), np.asarray(y)
        
        if shuffle:
            index = np.random.permutation(X.shape[0])
            X, y = X[index], y[index]
            
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches  = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.X[k * bs:(k+1) * bs], self.y[k * bs:(k+1) * bs]

def batches(X, y, bs=32, shuffle=True):
    for xb, yb in ReviewsIterator(X, y, bs, shuffle):
        xb = torch.LongTensor(xb)
        yb = torch.FloatTensor(yb)
        yield xb, yb.view(-1, 1).contiguous()

class EmbeddingNet(nn.Module):
    def __init__(self, n_users, n_movies, 
                 n_factors=50, embedding_dropout=0.02,
                 hidden=10, dropouts=0.2):
        super().__init__()
        hidden = get_list(hidden)
        dropouts = get_list(dropouts)
        n_last = hidden[-1]
        
        def gen_layers(n_in):
#             生成隐藏层的辅助函数
            nonlocal hidden, dropouts
            assert len(dropouts) <= len(hidden)
            for n_out, rate in zip_longest(hidden, dropouts):
                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()
                if rate is not None and rate > 0:
                    yield nn.Dropout(rate)
                n_in = n_out
        
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden = nn.Sequential(*list(gen_layers(2*n_factors)))
        self.fc = nn.Linear(n_last, 1)
        self._init()

    def forward(self, users, movies, minmax=None):
        features = torch.cat([self.u(users), self.m(movies)], dim=1)
        x = self.drop(features)
        x = self.hidden(x)
        out = torch.sigmoid(self.fc(x))
        if minmax is not None:
            min_rating, max_rating = minmax
            out = out * (max_rating - min_rating + 1) + min_rating - 0.5
        return out 
    
    def _init(self):
        pass
#         def init(m):
#             if type(m) == nn.Linear:
#                 torch.nn.init.xavier_uniform_(m.weight)
#                 m.bias.data.fill_(0.001)
                        
#         self.u.weight.data.uniform_(-0.05, 0.05)
#         self.m.weight.data.uniform_(-0.05, 0.05)
        
#         self.hidden.apply(init)
#         init(self.fc)

def get_list(n):
    if isinstance(n, (int, float)):
        return [n]
    elif hasattr(n, '__iter__'):
        return list(n)
    raise TypeError('layers configuration should be a single number or a list of numbers')

device = torch.device('cuda:0')

net = EmbeddingNet(n_users=n, n_movies=m,
                  n_factors=150, hidden=[500, 500, 500],
                  embedding_dropout=0.05, dropouts=[0.5, 0.5, 0.25]).to(device)