import matplotlib.pyplot as plt
from nasa_data_tool import getdata
import numpy as np
#%%
# all_charge_records,all_discharge_records,charge_length,discharge_length = getdata(
#     path = r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat",
#     squence_length = 150,   # 充电都建议100的长度，但是充电采样1min/次，放电1s/次
#     expand_multiple = 10,
#     output_type=["C","D"]
#     )
# plt.hist(charge_length)
# plt.show()
# plt.hist(discharge_length)
# plt.show()
# print(all_discharge_records[2].capacity)    # 用于计算SOH
# print(all_discharge_records[2].data.shape)  # 数据，内部结构见下
# print(all_discharge_records[2].type)        # 充电C 放电D
# print("load success")

# data(3,500)的内容
# voltage          电压，噪声+-2%，偏移+-0.1V
# current          电流，噪声+-2%，偏移+-0.005A
# temperature      温度，噪声+-2%，偏移+-2.5C

# NOTE: 现在charge的output还是6秒一次取样

#%% 统计随即充放电的数据长度
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
    r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat",
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
for discharge_data in all_discharge_data:
    discharge_data.data = np.stack((
                                    (discharge_data.data[0] - discharge_range[0])/(discharge_range[3]-discharge_range[0]),
                                    (discharge_data.data[1] - discharge_range[1])/(discharge_range[4]-discharge_range[1]),
                                    (discharge_data.data[2] - discharge_range[2])/(discharge_range[5]-discharge_range[2]),
                                ),axis=0)

print("done")