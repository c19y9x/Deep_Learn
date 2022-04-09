import torch
import torchvision
from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd
import utils

print(utils.train_valid_split(pd.read_csv('../data/covid.train.csv').values,0.3))
# 读取数据集
