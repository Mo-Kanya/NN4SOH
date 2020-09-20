from nasa_data_tool import get_data
rw_cycles = get_data(r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat",verbose=True)

print(rw_cycles[2].capacity)                # 用于计算SOH
print(rw_cycles[2].merged_discharge.shape)  # 放电，内部结构见下
print(rw_cycles[2].charge.shape)            # 充电，内部结构见下


# 
# time             绝对时间 （用不到）
# relativeTime     相对时间 （用不到）
# voltage          电压，噪声+-2%，偏移+-0.1V
# current          电流，噪声+-2%，偏移+-0.005A
# temperature      温度，噪声+-2%，偏移+-2.5C