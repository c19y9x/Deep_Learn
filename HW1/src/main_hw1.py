import torch
import torchvision
from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd
import utils

# print(utils.train_valid_split(pd.read_csv('../data/covid.train.csv').values,0.3))
df = pd.read_csv('../data/covid.train.csv')
data = df.values
for i in range(0,3):
    feature_data = data[:,102-16*i:117-16*i]
    label_data = data[:,-1-16*i]
    column = list(df.columns.values)[102-16*i:117-16*i]
    print(column)
    print(utils.get_feature_importance(feature_data,label_data.astype("int"),9,column))
# 读取数据集
