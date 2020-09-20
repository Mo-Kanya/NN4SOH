from nasa_data_tool import getdata
all_charge_records,all_discharge_records = getdata(
    path = r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat",
    squence_length = 500,
    expand_multiple = 10
    )

print(all_discharge_records[2].capacity)    # 用于计算SOH
print(all_discharge_records[2].data.shape)  # 数据，内部结构见下
print(all_discharge_records[2].type)        # 充电C 放电D
print("load success")

# data(3,500)的内容
# voltage          电压，噪声+-2%，偏移+-0.1V
# current          电流，噪声+-2%，偏移+-0.005A
# temperature      温度，噪声+-2%，偏移+-2.5C