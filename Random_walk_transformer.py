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
# data = getdata(path=r'/Users/jason/NN4SOH/dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat')[1]
path_list = [
    # r"./dataset/nasa/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW1.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW2.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW7.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW8.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW3.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW4.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW5.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW6.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW9.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW10.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW11.mat",
    # r"./dataset/nasa/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW12.mat",
    r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW13.mat",
    r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW14.mat",
    r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW15.mat",
    r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW16.mat",
    r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW17.mat",
    r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW18.mat",
    r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW19.mat",
    r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW20.mat",
    r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW21.mat",
    r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat",
    r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW23.mat",
    r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW24.mat",
    r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW25.mat",
    r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW26.mat",
    r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW27.mat",
    r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW28.mat",
]

all_charge_data = []
all_discharge_data = []
for path in path_list: 
    charge_data,discharge_data,_,_ = getdata(
        path = path,
        squence_length = 100,
        expand_multiple = 4,
        output_type=["C","D"]
        )
    all_charge_data += charge_data
    all_discharge_data += discharge_data
data = all_discharge_data

dt = []
labels = []
for i in range(7000):
    dt.append(data[i].data.T)
    labels.append(data[i].SOH)

mm = MinMaxScaler()
for i in range(len(dt)):
    mm = MinMaxScaler()
    dt[i] = mm.fit_transform(dt[i])
dt,labels = np.array(dt),np.array(labels)
print(dt.shape)
data=torch.from_numpy(dt).type(torch.FloatTensor)
labels=torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

x_train = data[:5000]
x_val = data[5000: 6000]
x_test = data[6000:]
y_train = labels[:5000]
y_val = labels[5000: 6000]
y_test = labels[6000:]
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_ds, batch_size=4)
val_loader = DataLoader(val_ds, batch_size=4)
test_loader = DataLoader(test_ds, batch_size=4)

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
seq_len = 100
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
