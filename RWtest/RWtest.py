import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nasa_data_tool import getdata
from bilstm import BiLSTM
from gru import GRU

import torch
import torch.nn as nn
from torch.autograd import Variable
# path_list = ["/home/mokanya/RWtest/RW22.mat"]
path_list = ["./RW22.mat"]
all_charge_data = []
all_discharge_data = []
inf=999999999.0
# discharge_range = (min_V,min_C,min_T,max_V,max_C,max_T )
discharge_range =   [inf, inf,   inf,  -inf, -inf,  -inf]
for path in path_list:
    charge_data,discharge_data,_,_ = getdata(
        path = path,
        squence_length = 100,
        expand_multiple = 4,
        output_type=["C","D"]
        )
    all_discharge_data += discharge_data
for discharge_data in all_discharge_data:
    min_V,min_C,min_T = discharge_data.data.min(axis=1)
    max_V,max_C,max_T = discharge_data.data.max(axis=1)
    discharge_range[0] = discharge_range[0] if discharge_range[0] < min_V else min_V
    discharge_range[1] = discharge_range[1] if discharge_range[1] < min_C else min_C
    discharge_range[2] = discharge_range[2] if discharge_range[2] < min_T else min_T
    discharge_range[3] = discharge_range[3] if discharge_range[3] > max_V else max_V
    discharge_range[4] = discharge_range[4] if discharge_range[4] > max_C else max_C
    discharge_range[5] = discharge_range[5] if discharge_range[5] > max_T else max_T
for discharge_data in all_discharge_data:
    discharge_data.data = np.stack((
                                    (discharge_data.data[0] - discharge_range[0])/(discharge_range[3]-discharge_range[0]),
                                    (discharge_data.data[1] - discharge_range[1])/(discharge_range[4]-discharge_range[1]),
                                    (discharge_data.data[2] - discharge_range[2])/(discharge_range[5]-discharge_range[2]),
                                ),axis=0)
train_set = []
valid_set = []
test_set = []
for i in range(14000):
    if len(all_discharge_data[i].data[0]) == 100:
        train_set.append(all_discharge_data[i])
for i in range(14000, 18000):
    if len(all_discharge_data[i].data[0]) == 100:
        valid_set.append(all_discharge_data[i])
for i in range(18000, 20000):
    if len(all_discharge_data[i].data[0]) == 100:
        test_set.append(all_discharge_data[i])

shapes = [(16, 1, (2, 3)), (32, 1, (1,2,3)), (64, 1, (1,2,3)),
              (32, 2, (1,2,3)), (64, 2, (1,2,3))]
strategies = [(1, 0.0001, 40), (2, 0.00015, 30), (4, 0.0002, 24), (8, 0.0003, 18)]

f = open("log_gru.txt", "a")

for shape in shapes:
    for option in shape[2]:
        strategy = strategies[option]
        rnn = GRU(3, shape[0], shape[1], 1)
        rnn.cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(rnn.parameters(), lr=strategy[1])
        for epoch in range(strategy[2]):
            for i in range(int(len(train_set) / strategy[0])):
                img = []
                lb = []
                for j in range(strategy[0]):
                    img.append(np.array(train_set[i*strategy[0]+j].data, dtype=np.float).transpose())
                    lb.append(np.array(train_set[i*strategy[0]+j].SOH, dtype=np.float))
                img = np.array(img)
                img = torch.FloatTensor(img)
                img = Variable(img).cuda()
                lb = np.array(lb)
                lb = torch.FloatTensor(lb)
                lb = Variable(lb).cuda()

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                output = rnn(img)
                loss = criterion(output, lb)
                loss.backward()
                optimizer.step()
        xs = []
        ys = []
        for k in range(len(valid_set)):
            xs.append(np.array(valid_set[k].data, dtype=np.float).transpose())
            ys.append(np.array(train_set[k].SOH, dtype=np.float))
        xs = np.array(xs)
        xs = torch.FloatTensor(xs)
        xs = Variable(xs).cuda()
        ys = np.array(ys)
        ys = torch.FloatTensor(ys)
        ys = Variable(ys).cuda()
        yc = rnn(xs)
        f.write('\ntype: BiLSTM\thid_size: {}\tnum_layers: {}\tbatch_size: {}\tlr: {}\tepochs:{}\terror: {}\n'.format( shape[0], shape[1], strategy[0], strategy[1], strategy[2], criterion(ys, yc).item() ))
        f.flush()

f.close()
