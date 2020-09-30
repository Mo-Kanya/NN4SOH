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

    # r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW13.mat",
    # r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW14.mat",
    # r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW15.mat",
    # r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW16.mat",
    # r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW17.mat",
    # r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW18.mat",
    # r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW19.mat",
    # r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW20.mat",
    # r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW21.mat",
    r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat",
    # r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW23.mat",
    # r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW24.mat",
    # r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW25.mat",
    # r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW26.mat",
    # r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW27.mat",
    # r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW28.mat",
]

# all_charge_length=[]
# all_discharge_length=[]
# Cfig,Caxes = plt.subplots(4,4)
# Dfig,Daxes = plt.subplots(4,4)
# counter = 0
# for path in path_list: 
#     _,_,charge_length,discharge_length = getdata(
#         path = path,
#         squence_length = 1000,
#         expand_multiple = 0,
#         output_type=["C","D"]
#         )
#     all_charge_length += charge_length
#     all_discharge_length += discharge_length

#     Cax = Caxes[counter//4,counter%4]
#     Cax.hist(charge_length)
#     Cax.set_title(path.split('/')[-1])
#     Cax.set_xlim(0,350)

#     Dax = Daxes[counter//4,counter%4]
#     Dax.hist(discharge_length)
#     Dax.set_title(path.split('/')[-1])
#     Dax.set_xlim(0,5000)
#     counter +=1
# plt.show()
# # plt.hist(all_charge_length)
# # plt.show()
# # plt.hist(all_discharge_length)
# # plt.show()

# 用一下部分替换原来提取数据集的部分
path_list = [
    # r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW13.mat",
    # r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW14.mat",
    # r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW15.mat",
    # r"./dataset/nasa/RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW16.mat",
    # r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW17.mat",
    # r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW18.mat",
    # r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW19.mat",
    # r"./dataset/nasa/RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW20.mat",
    # r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW21.mat",
    r"/Users/jason/NN4SOH/dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat",
    # r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW23.mat",
    # r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW24.mat",
    # r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW25.mat",
    # r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW26.mat",
    # r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW27.mat",
    # r"./dataset/nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW28.mat",
]
all_charge_data = []
all_discharge_data = []
inf=999999999.0
# charge_range    = (min_V,min_C,min_T,max_V,max_C,max_T )   # max,min
charge_range    =   [inf, inf,   inf,  -inf, -inf,  -inf]   # max,min
# discharge_range = (min_V,min_C,min_T,max_V,max_C,max_T )
discharge_range =   [inf, inf,   inf,  -inf, -inf,  -inf] 
for path in path_list: 
    charge_data,discharge_data,_,_ = getdata(
        path = path,
        squence_length = 100,
        expand_multiple = 4,
        output_type=["C","D"]
        )
    all_charge_data += charge_data
    all_discharge_data += discharge_data
for charge_data in all_charge_data:
    min_V,min_C,min_T = charge_data.data.min(axis=1)
    max_V,max_C,max_T = charge_data.data.max(axis=1)
    charge_range[0] = charge_range[0] if charge_range[0] < min_V else min_V
    charge_range[1] = charge_range[1] if charge_range[1] < min_C else min_C
    charge_range[2] = charge_range[2] if charge_range[2] < min_T else min_T
    charge_range[3] = charge_range[3] if charge_range[3] > max_V else max_V
    charge_range[4] = charge_range[4] if charge_range[4] > max_C else max_C
    charge_range[5] = charge_range[5] if charge_range[5] > max_T else max_T
for charge_data in all_charge_data:
    charge_data.data = np.stack((
                                    (charge_data.data[0] - charge_range[0])/(charge_range[3]-charge_range[0]),
                                    (charge_data.data[1] - charge_range[1])/(charge_range[4]-charge_range[1]),
                                    (charge_data.data[2] - charge_range[2])/(charge_range[5]-charge_range[2]),
                                ),axis=0)

for discharge_data in all_discharge_data:
    min_V,min_C,min_T = discharge_data.data.min(axis=1)
    max_V,max_C,max_T = discharge_data.data.max(axis=1)
    discharge_range[0] = discharge_range[0] if discharge_range[0] < min_V else min_V
    discharge_range[1] = discharge_range[1] if discharge_range[1] < min_C else min_C
    discharge_range[2] = discharge_range[2] if discharge_range[2] < min_T else min_T
    discharge_range[3] = discharge_range[3] if discharge_range[3] > max_V else max_V
    discharge_range[4] = discharge_range[4] if discharge_range[4] > max_C else max_C
    discharge_range[5] = discharge_range[5] if discharge_range[5] > max_T else max_T
dt=[]
for discharge_data in all_discharge_data:
    discharge_data.data = np.stack((
                                    (discharge_data.data[0] - discharge_range[0])/(discharge_range[3]-discharge_range[0]),
                                    (discharge_data.data[1] - discharge_range[1])/(discharge_range[4]-discharge_range[1]),
                                    (discharge_data.data[2] - discharge_range[2])/(discharge_range[5]-discharge_range[2]),
                                ),axis=0)
    dt.append(discharge_data.data)


x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
for i in range(14000):
    if len(all_discharge_data[i].data[0]) == 100:
        x_train.append(all_discharge_data[i].data.T)
        y_train.append(all_discharge_data[i].SOH)
for i in range(14000, 18000):
    if len(all_discharge_data[i].data[0]) == 100:
        x_val.append(all_discharge_data[i].data.T)
        y_val.append(all_discharge_data[i].SOH)
for i in range(18000, 20000):
    if len(all_discharge_data[i].data[0]) == 100:
        x_test.append(all_discharge_data[i].data.T)
        y_test.append(all_discharge_data[i].SOH)


x_train=torch.from_numpy(np.array(x_train)).type(torch.FloatTensor)
y_train=torch.from_numpy(np.array(y_train)).type(torch.FloatTensor)
x_val=torch.from_numpy(np.array(x_val)).type(torch.FloatTensor)
y_val=torch.from_numpy(np.array(y_val)).type(torch.FloatTensor)
x_test=torch.from_numpy(np.array(x_test)).type(torch.FloatTensor)
y_test=torch.from_numpy(np.array(y_test)).type(torch.FloatTensor)

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
