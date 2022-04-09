# 用到的一些类和函数
import torch
from torch.utils.data import Dataset, random_split

import pandas as pd
import numpy as np

def train_valid_split(data_set, valid_ratio):
    '''将提供的培训数据分为培训集和验证集'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(41))
    return np.array(train_set), np.array(valid_set)

def select_feature(train_set,valid_set,test_set,select_all=True):
    '''Selects useful features to perform regression
        选择有用的特征以执行回归
    '''
    y_train, y_valid = train_set[:, -1], valid_set[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_set[:, :-1], valid_set[:, :-1], test_set

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]  # TODO: Select suitable feature columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

class COVID19Dataset(Dataset):
    def __init__(self,x,y=None):
        if y==None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
# 测试
# COVID19Dataset('../data/covid.train.csv')