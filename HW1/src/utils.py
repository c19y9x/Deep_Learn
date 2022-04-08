# 用到的一些类和函数

from torch.utils.data import Dataset

import pandas as pd

class COVID19Dataset(Dataset):
    def __init__(self,train_path,test_path):
        train_data,test_data = pd.read_csv(train_path).values, pd.read_csv(test_path).values


    def __getitem__(self, idx):
        return

    def __len__(self):
        return