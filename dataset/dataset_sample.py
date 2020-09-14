# coding:UTF-8

#%%
import scipy.io as scio
import numpy as np

#%%
dataFile = "./nasa/BatteryAgingARC_25_26_27_28_P1/B0025.mat"
data = scio.loadmat(dataFile)


# %%
print(data.keys())  # ['__header__', '__version__', '__globals__', 'B0025'])
print(type(data["B0025"]))  # <class 'numpy.ndarray'>
print(data["__header__"])
print(data["__version__"])
print(data["__globals__"])

#%%

print(data["B0025"][0].shape)
print(data["B0025"][0].dtype)
print(data["B0025"][0][0].shape)
print(data["B0025"][0][0].dtype)
print(data["B0025"][0][0][0].shape)
print(data["B0025"][0][0][0].dtype)
print(data["B0025"][0][0][0][0].shape)
print(data["B0025"][0][0][0][0].dtype)
print(data["B0025"][0][0][0][0][0].shape)
print(data["B0025"][0][0][0][0][0].dtype)

#%%
print(data["B0025"][0][0][0][0][0][3].shape)
print(data["B0025"][0][0][0][0][0][3].dtype)
print(data["B0025"][0][0][0][0][0][3])
#%%
print(data["B0025"][0][0][0][0][0][3][0].shape)
print(data["B0025"][0][0][0][0][0][3][0].dtype)
# print(data["B0025"][0][0][0][0][0][3][0])
#%%
print(data["B0025"][0][0][0][0][0])


# %%
