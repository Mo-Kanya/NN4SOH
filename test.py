"""
Other ARC-FY data files can be parsed with slight modification based on their own specific situation
REMEMBER TO MODIFY THE PATH BEFORE RUNNING!
"""
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from bilstm import BiLSTM
from gru import GRU

import torch
import torch.nn as nn
from torch.autograd import Variable

dataFile = '/Users/kanya/Projects/NN4SOH/dataset/ARC-FY/B0005'   # Modify this path
raw = scio.loadmat(dataFile)['B0005'][0][0][0][0]

# raw data parsing
cycles = []
labels = []
for i in range(len(raw)):
    if raw[i][0] == ['charge']:
        if i+1 != len(raw) and raw[i+1][0] != ['charge'] and len(raw[i][3][0][0][0][0]) > 850: # discard unfair records
            cycles.append(raw[i][3][0][0])
            if raw[i+1][0] == ['discharge']:
                labels.append(raw[i+1][3][0][0][6][0])
            elif i+2 != len(raw) and raw[i+2][0] == ['discharge']:
                labels.append(raw[i+2][3][0][0][6][0])
assert (len(cycles) == len(labels)), 'Number of measurements not matched!'

data = []
# calculate SOHs
for lb in range(len(labels)):
    labels[lb] = labels[lb][0] / 1.856487420818157  # TODO: first (largest) capacity found, but probably not the full cp
labels = labels * 3

for t0 in [0, 1.5, 3]:
    for cy in cycles:
        t = t0
        t_limit = 4000 + t0  # TODO: this parameter can be further tuned
        cursor = 0
        cy_new = []
        while cursor <= len(cy[0][0]) and t <= t_limit:
            while cy[5][0][cursor] <= t:
                cursor += 1
            x1 = cy[5][0][cursor - 1]
            x2 = cy[5][0][cursor]
            point = []
            for i in range(3):
                y1 = cy[i][0][cursor - 1]
                y2 = cy[i][0][cursor]
                y = (t - x1) * (y2 - y1) / (x2 - x1) + y1
                point.append(y)
            cy_new.append(point)
            cursor -= 1
            t += 10
        data.append(cy_new)

for i in range(len(data)):
    mm = MinMaxScaler()
    data[i] = mm.fit_transform(data[i])

data_set = list(zip(data, labels))
np.random.shuffle(data_set)
train_set = data_set[:400]
valid_set = data_set[400: 450]
test_set = data_set[450:]
for v in np.random.randint(0, 400, 25):
    valid_set.append(data_set[v])
for t in np.random.randint(0, 450, 30):
    test_set.append(data_set[t])

sequence_length = 401
input_size = 3
hidden_size = 16
num_layers = 1
batch_size = 1
num_epochs = 2
learning_rate = 0.0001

for ktr in range(1):
    rnn = BiLSTM(input_size, hidden_size, num_layers, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    losses = []
    ktr_in = 0
    for epoch in range(num_epochs):
        for img, lb in train_set:
            img = np.array([img,],dtype=np.float)
            img = torch.FloatTensor(img)
            img = Variable(img)  # .cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            output = rnn(img)
            loss = criterion(output, torch.FloatTensor(np.array([[lb,],], dtype=np.float)))
            loss.backward()
            optimizer.step()
            if ktr_in % 10 == 0:
                losses.append(loss)
            ktr_in += 10
    plt.plot(losses)
    plt.show()
