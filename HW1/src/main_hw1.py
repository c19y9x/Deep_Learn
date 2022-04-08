import torchvision
from torch.utils.data import Dataset

import pandas as pd

class COVID19Dataset(Dataset):
    def __init__(self,train_path,test_path):
        train_data,test_data = pd.read_csv(train_path).values, pd.read_csv(test_path).values
        print(type(train_data),type(test_data))

    def __getitem__(self, idx):
        return

    def __len__(self):
        return
# 测试
COVID19Dataset('../data/covid.train.csv','../data/covid.test.csv')

# 读取数据集
