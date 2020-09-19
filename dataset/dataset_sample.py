# coding:UTF-8
#%% 导入模块
import scipy.io as scio
import numpy as np

# %% [markdown] 随机放电 RW
# # 随机放电 RW

#%% 导入文件 
dataFile = r"./nasa/RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW27.mat"
data = scio.loadmat(dataFile)

#%% 随机过程 uniform
# print(data.keys())  # ['__header__', '__version__', '__globals__', 'data'])
# print(data["data"].dtype)  # ('step', 'O'), ('procedure', 'O'), ('description', 'O')
# print(data["data"]["step"].dtype)   # object
# print(data["data"]["step"].shape)   # (1,1)
print(data["data"]["step"][0, 0].dtype)     # [('comment', 'O'), ('type', 'O'), ('time', 'O'), ('relativeTime', 'O'), ('voltage', 'O'), ('current', 'O'), ('temperature', 'O'), ('date', 'O')]
print(data["data"]["step"][0, 0].shape)     # (1,23659)
print(np.unique(data["data"]["step"][0, 0]['comment']))    # (1,)
#%%

#     data["data"]["step"][0, 0]['attribute']   [0,num_of_cycle].shape
i = num_of_cycle = 0 # 根据时间排序的
print(data["data"]["step"][0, 0]['comment']     [0,i].shape)    # (1,)
print(data["data"]["step"][0, 0]['type']        [0,i].shape)    # (1,)
print(data["data"]["step"][0, 0]['relativeTime'][0,i].shape)    # (1, 176)
print(data["data"]["step"][0, 0]['voltage']     [0,i].shape)    # (1, 176)
print(data["data"]["step"][0, 0]['current']     [0,i].shape)    # (1, 176)
print(data["data"]["step"][0, 0]['temperature'] [0,i].shape)    # (1, 176)
print(data["data"]["step"][0, 0]['date']        [0,i].shape)    # (1,)

print(data["data"]["step"][0, 0]['relativeTime'][0,i])          # (1, 176)
print(data["data"]["step"][0, 0]['voltage']     [0,i])          # (1, 176)
print(data["data"]["step"][0, 0]['current']     [0,i])          # (1, 176)
print(data["data"]["step"][0, 0]['temperature'] [0,i])          # (1, 176)

# %% [markdown] 随机充放电 RW
# # 随机充放电 RW

#%% 导入文件 
dataFile = r"./nasa/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab//RW7.mat"
data = scio.loadmat(dataFile)

#%% 随机过程 uniform
print(data.keys())  # ['__header__', '__version__', '__globals__', 'data'])
print(data["data"].dtype)  # ('step', 'O'), ('procedure', 'O'), ('description', 'O')
print(data["data"]["step"].dtype)   # object
print(data["data"]["step"].shape)   # (1,1)
print(data["data"]["step"][0, 0].dtype)     # [('comment', 'O'), ('type', 'O'), ('time', 'O'), ('relativeTime', 'O'), ('voltage', 'O'), ('current', 'O'), ('temperature', 'O'), ('date', 'O')]
print(data["data"]["step"][0, 0].shape)     # (1,17277)
print(data["data"]["step"][0, 0]["comment"].dtype)  # object
print(data["data"]["step"][0, 0]["comment"].shape)  # (1, 17277)
print(np.unique(data["data"]["step"][0, 0]["comment"])) # cycle 的类型

# %% [markdown] 非随机过程
# # 非随机过程

#%% 导入文件 
dataFile = r"./nasa/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab//RW7.mat"
data = scio.loadmat(dataFile)

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
