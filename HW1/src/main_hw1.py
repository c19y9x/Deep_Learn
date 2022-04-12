import torch
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import utils

# print(utils.train_valid_split(pd.read_csv('../data/covid.train.csv').values,0.3))
df = pd.read_csv('../data/covid.train.csv')
df1 = pd.read_csv('../data/test.csv')
# 训练验证集
data = df.values
# 测试集
test_data = df1.values
# for i in range(0,3):
#     feature_data = data[:,102-16*i:117-16*i]
#     label_data = data[:,-1-16*i]
#     column = list(df.columns.values)[102-16*i:117-16*i]
#     print(column)
#     print(utils.get_feature_importance(feature_data,label_data.astype("int"),10,column))

# 读取数据集

# 划分数据集为验证集和训练集
train_data , valid_data = utils.train_valid_split(data, 0.2)

# print(train_data.shape,valid_data.shape)
# 挑选特征

# feature_data = data[:,102:117]
# label_data = data[:,-1]
# column = list(df.columns.values)[102:117]
# # 获得特征列名称
# indices = utils.get_feature_importance(feature_data,label_data.astype("int"),10,column)
#
# indices = 102+indices
# indices1 = []
# for i in range(0,5):
#     indices2 = indices-16
#     indices1 = np.append(indices1,indices2)
#     indices = indices-16
# indices = indices1+16
# indices = indices.astype("int")

train_label_data , valid_label_data = train_data[:,-1] , valid_data[:,-1]
train_feature_data , valid_feature_data = train_data[:,1:-1],valid_data[:,1:-1]
test_data = test_data[:,:-1]
print(train_feature_data.shape,valid_feature_data.shape,train_label_data.shape,valid_label_data.shape,test_data.shape)
train_feature_data[:,37:] = (train_feature_data[:,37:] - train_feature_data[:,37:].mean(axis=0, keepdims=True))/train_feature_data[:,37:].std(axis=0, keepdims=True)
valid_feature_data[:,37:] = (valid_feature_data[:,37:] - valid_feature_data[:,37:].mean(axis=0, keepdims=True))/valid_feature_data[:,37:].std(axis=0, keepdims=True)


train_dataset = utils.COVID19Dataset(train_feature_data,train_label_data)
valid_dataset = utils.COVID19Dataset(valid_feature_data,valid_label_data)
test_dataset = utils.COVID19Dataset(valid_data)


# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True,drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True, pin_memory=True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, pin_memory=True,drop_last=True)

myModel = utils.My_Model(train_feature_data.shape[1])
if torch.cuda.is_available():
    myModel = myModel.cuda()

loss1 = nn.MSELoss()
if torch.cuda.is_available():
    loss1 = loss1.cuda()

learning_rate = 1e-3
optimizer = torch.optim.SGD(myModel.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 3000

# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    myModel.train()
    for data in train_loader:
        x, y = data
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        outputs = myModel(x)
        loss = loss1(outputs, y)
        # print(loss)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # if total_train_step % 100 == 0:
        print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    myModel.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in valid_loader:
            x,y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            outputs = myModel(x)
            loss = loss1(outputs, y)
            total_test_loss = total_test_loss + loss.item()
            # accuracy = (outputs.argmax(1) == targets).sum()
            # total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    # print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

# test_data = torch.FloatTensor(test_data)
# test_data = (test_data - test_data.mean(dim=0, keepdim=True))/test_data.std(dim=0, keepdim=True)
# if torch.cuda.is_available():
#     print(myModel(test_data.cuda()))
# else:
#     print(myModel(test_data))
torch.save(myModel, "myModel1.pth")
print("模型已保存")
