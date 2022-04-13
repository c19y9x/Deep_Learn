import csv

import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader

import utils

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
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x


model = My_Model(116)
# model.load_state_dict(torch.load("./models/model1.ckpt"))
model = torch.load("myModel1.pth")
model.cuda()
test_data = pd.read_csv('../data/covid.test.csv').values[:,1:]
test_dataset = utils.COVID19Dataset(test_data)
testLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,drop_last=True)

loss1 = nn.MSELoss()
val_rel = []
with torch.no_grad():
    for data in testLoader:
        if torch.cuda.is_available():
            x = data
            if torch.cuda.is_available():
                x = x.cuda()
            pred = model(x)
            val_rel.append(pred.item())
        else:
            pred = model(x)
            val_rel.append(pred.item())
with open('../data/test1.csv', 'w') as f:
    csv_writer = csv.writer(f)  # 百度的csv写法
    csv_writer.writerow(['id','tested_positive'])
    for i in range(len(test_data)):
        csv_writer.writerow([str(i),str(val_rel[i])])
