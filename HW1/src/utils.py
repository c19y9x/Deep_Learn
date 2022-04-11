# 用到的一些类和函数
import torch
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from torch import nn
from torch.utils.data import Dataset, random_split

import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter


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
        feat_idx = [0, 1, 2, 3, 4]  # Select suitable feature columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

class COVID19Dataset(Dataset):
    def __init__(self,x,y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)
        # self.x = (self.x - self.x.mean(dim=0, keepdim=True))/self.x.std(dim=0, keepdim=True)
    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
def get_feature_importance(feature_data, label_data, k =4,column = None):
    """
    column为列名
    """
    model = SelectKBest(chi2, k=k)#选择k个最佳特征
    X_new = model.fit_transform(feature_data, label_data)
    #feature_data是特征数据，label_data是标签数据，该函数可以选择出k个特征
    # print('x_new', X_new)
    scores = model.scores_
    # print(scores)
    # 按重要性排序，选出最重要的 k 个
    indices = np.argsort(scores)[::-1] #找到重要K个的下标
    if column:
        k_best_features = [column[i] for i in indices[0:k].tolist()]
        # print('k best features are: ',k_best_features)
    # return X_new, indices[0:k]
    return indices[0:k]

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        y = self.layers(x)
        y = y.squeeze(1) # (B, 1) -> (B) 如：[[3],[2],[1]]->[3,2,1]
        return y


# 测试
# COVID19Dataset('../data/covid.train.csv')