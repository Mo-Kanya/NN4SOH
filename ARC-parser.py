"""
Other ARC-FY data files can be parsed with slight modification based on their own specific situation
REMEMBER TO MODIFY THE PATH BEFORE RUNNING!
"""
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

# plot part of the (voltage) data by order
# i = 0
# while i <= 165:
#     time = np.linspace(0, 4000, 401)
#     voltage = []
#     for t in range(401):
#         voltage.append(data[i][t][0])
#     plt.plot(time, voltage)
#     plt.show()
#     i += 10

# Expand dataset by changing the starting point

# shape: (495, 401, 3)

for i in range(len(data)):
    mm = MinMaxScaler()
    data[i] = mm.fit_transform(data[i])

train_set = data[:400]
valid_set = data[400: 450]
test_set = data[450:]
for v in np.random.randint(0, 400, 25):
    valid_set.append(data[v])
for t in np.random.randint(0, 450, 30):
    test_set.append(data[t])
