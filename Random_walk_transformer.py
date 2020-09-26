from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nasa_data_tool import getdata

from SAnD.core.model import SAnD
from SAnD.utils.trainer import NeuralNetworkClassifier

# Real Dataset Generator
# dataFile = '/home/zhengshen/NN4SOH/dataset/ARC-FY/B0005'   # Modify this path
# dataFile = '/Users/jason/NN4SOH/dataset/ARC-FY/B0005'
data = getdata()[1]
train_set = []
valid_set = []
test_set = []
for i in range(5000):
    if len(data[i].data[0]) == 500:
        train_set.append(data[i])
for i in range(5000, 6000):
    if len(data[i].data[0]) == 500:
        valid_set.append(data[i])
for i in range(6000, 7000):
    if len(data[i].data[0]) == 500:
        test_set.append(data[i])

for i in range(len(data)):
    mm = MinMaxScaler()
    data[i] = mm.fit_transform(data[i])
data = np.array(data)
data = data[: , 0:400 ,:]
print(data.shape)
data=torch.from_numpy(data).type(torch.FloatTensor)
labels=torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

# data_set = list(zip(data, labels))
# np.random.shuffle(data_set)
# data, labels = data_set[0], data_set[1]
x_train = data[:400]
x_val = data[400: 450]
x_test = data[450:]
y_train = labels[:400]
y_val = labels[400: 450]
y_test = labels[450:]
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_ds, batch_size=1)
val_loader = DataLoader(val_ds, batch_size=1)
test_loader = DataLoader(test_ds, batch_size=1)

# Fake Dataset Generater
# x_train = torch.randn(1024, 256, 23)    # [N, seq_len, features]
# x_val = torch.randn(128, 256, 23)       # [N, seq_len, features]
# x_test =  torch.randn(512, 256, 23)     # [N, seq_len, features]

# y_train = torch.randint(0, 9, (1024, ))
# y_val = torch.randint(0, 9, (128, ))
# y_test = torch.randint(0, 9, (512, ))


# train_ds = TensorDataset(x_train, y_train)
# val_ds = TensorDataset(x_val, y_val)
# test_ds = TensorDataset(x_test, y_test)

# train_loader = DataLoader(train_ds, batch_size=128)
# val_loader = DataLoader(val_ds, batch_size=128)
# test_loader = DataLoader(test_ds, batch_size=128)

# Training
in_feature = 3
seq_len = 400
n_heads = 32
factor = 32
num_class = 1
num_layers = 4

clf = NeuralNetworkClassifier(
    SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    nn.MSELoss(),
    optim.Adam, optimizer_config={"lr": 1e-4, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
    experiment=Experiment("8mKGHiYeg2P7dZEFlvQv3PEzc")
)

# from SAnD.core.model import SAnD
# from SAnD.core.modules import RegressionModule


# class RegSAnD(SAnD):
#     def __init__(self, *args, **kwargs):
#         super(RegSAnD, self).__init__(*args, **kwargs)
#         d_model = kwargs.get("d_model")
#         factor = kwargs.get("factor")
#         output_size = kwargs.get("n_class")    # output_size

#         self.clf = RegressionModule(d_model, factor, output_size)


# # model = RegSAnD(
# #     input_features=..., seq_len=..., n_heads=..., factor=...,
# #     n_class=..., n_layers=...
# # )

# clf = NeuralNetworkClassifier(
#     RegSAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
#     nn.CrossEntropyLoss(),
#     optim.Adam, optimizer_config={"lr": 1e-5, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
#     experiment=Experiment("8mKGHiYeg2P7dZEFlvQv3PEzc")
# )

# training network
clf.fit(
    {"train": train_loader,
     "val": val_loader,
     "test": test_loader},
    epochs=80
)

# evaluating
# clf.evaluate(test_loader)

# save
clf.save_to_file("save_params/")
