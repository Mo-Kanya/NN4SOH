    # coding:UTF-8
def getdata(path=None,squence_length = 500,expand_multiple = 4,output_type=["C","D"]):
    #%% 导入模块
    from numpy.lib.function_base import average
    import scipy.io as scio
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import interpolate
    import random
    # %% [markdown] 随机放电 RW
    # # 随机放电 RW

    #%% 导入文件 
    if path:
        dataFile = path
    else:
        dataFile = r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat"
        # dataFile = r"./nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat"
    data = scio.loadmat(dataFile)

    #%% 读取，预览数据
    # # print(data.keys())  # ['__header__', '__version__', '__globals__', 'data'])
    # # print(data["data"].dtype)  # ('step', 'O'), ('procedure', 'O'), ('description', 'O')
    # # print(data["data"]["step"].dtype)   # object
    # # print(data["data"]["step"].shape)   # (1,1)
    # print(data["data"]["step"][0, 0].dtype)     
    # # [('comment', 'O'), ('type', 'O'), ('time', 'O'), ('relativeTime', 'O'), ('voltage', 'O'), ('current', 'O'), ('temperature', 'O'), ('date', 'O')]
    # print(data["data"]["step"][0, 0].shape)     # (1,23659)
    # print(np.unique(data["data"]["step"][0, 0]['comment']))    # 数据的类型
    # i = num_of_cycle = 0 # 根据时间排序的
    # #     data["data"]["step"][0, 0]['attribute']   [0,num_of_cycle].shape
    # print(data["data"]["step"][0, 0]['comment']     [0,i].shape)    # (1,)
    # print(data["data"]["step"][0, 0]['type']        [0,i].shape)    # (1,)
    # print(data["data"]["step"][0, 0]['relativeTime'][0,i].shape)    # (1, 176)
    # print(data["data"]["step"][0, 0]['voltage']     [0,i].shape)    # (1, 176)
    # print(data["data"]["step"][0, 0]['current']     [0,i].shape)    # (1, 176)
    # print(data["data"]["step"][0, 0]['temperature'] [0,i].shape)    # (1, 176)
    # print(data["data"]["step"][0, 0]['date']        [0,i].shape)    # (1,)

    # print(data["data"]["step"][0, 0]['relativeTime'][0,i])          # (1, 176)
    # print(data["data"]["step"][0, 0]['voltage']     [0,i])          # (1, 176)
    # print(data["data"]["step"][0, 0]['current']     [0,i])          # (1, 176)
    # print(data["data"]["step"][0, 0]['temperature'] [0,i])          # (1, 176)

    # #%% 
    # plt.plot(data["data"]["step"][0, 0]['relativeTime'][0,172][0,:],
    #          data["data"]["step"][0, 0]['voltage']     [0,172][0,:],)
    # plt.show()
    # plt.plot(data["data"]["step"][0, 0]['relativeTime'][0,172][0,:],
    #          data["data"]["step"][0, 0]['current']     [0,172][0,:],)
    # plt.show()
    # plt.plot(data["data"]["step"][0, 0]['relativeTime'][0,172][0,:],
    #          data["data"]["step"][0, 0]['temperature'] [0,172][0,:],)
    # plt.show()

    #%% 操作类型和数据属性

    useful_comments = [
        'charge (after random walk discharge)',
        'discharge (random walk)',
        # 'pulsed load (discharge)',
        # 'pulsed load (rest)',
        # 'reference charge',
        # 'reference discharge',
        # 'reference power discharge',
        # 'rest (random walk)',
        # 'rest post pulsed load',
        # 'rest post random walk discharge',
        # 'rest post reference charge',
        # 'rest post reference discharge',
        # 'rest post reference power discharge',
        # 'rest prior reference discharge'
    ]

    # comment      = data["data"]["step"][0, 0]['comment']     [0,i][0]
    # type         = data["data"]["step"][0, 0]['type']        [0,i][0]
    # relativeTime = data["data"]["step"][0, 0]['relativeTime'][0,i][0,:]   # 最后一步索引得到一维数组(107,)，否则是(1,107)
    # voltage      = data["data"]["step"][0, 0]['voltage']     [0,i][0,:]   # 最后一步索引得到一维数组(107,)，否则是(1,107)
    # current      = data["data"]["step"][0, 0]['current']     [0,i][0,:]   # 最后一步索引得到一维数组(107,)，否则是(1,107)
    # temperature  = data["data"]["step"][0, 0]['temperature'] [0,i][0,:]   # 最后一步索引得到一维数组(107,)，否则是(1,107)
    # date         = data["data"]["step"][0, 0]['date']        [0,i][0]

    attributes = [  
    # 'comment',    # 没必要
    'time',
    # 'type',       # 没必要
    'relativeTime',
    'voltage',
    'current',
    'temperature',
    # 'date'        # 没必要
    ]


    #%% 数据信息提取模板
    def info():
        # 每个大cycle的记录
        ref_idx = 0
        cycle_record = {x[0]:{'num':0,
                                # 'value':{attribute:[] for attribute in attributes}
                            }for x in np.unique(data["data"]["step"][0, 0]['comment'])}
        # 全部信息的记录
        record = {x[0]:0 for x in np.unique(data["data"]["step"][0, 0]['comment'])}

        # 对每一条操作进行遍历
        for i in range(data["data"]["step"][0, 0].shape[1]):

            # 当前操作信息处理
            cur = data["data"]["step"][0, 0]['comment'][0,i][0]
            cycle_record[cur]['num']+=1
            record[cur]+=1

            # 如果是有用的操作，则记录下这条操作的所有信息
            if cur == useful_comments:
                for attribute in attributes:
                    cycle_record[ref_idx][cur]['value'][attribute].append(data["data"]["step"][0,0][attribute][0,i][0,:])

            # 如果出现 'rest prior reference discharge'（大循环开始标志），则处理上一个大循环的信息，并重置大循环记录 
            if cur == 'rest prior reference discharge':
                # 打印当前大循环记录的信息
                print('\n'+'='*20+f'cycle {ref_idx}'+'='*20)
                for x in cycle_record:
                    print(f'{x}:{cycle_record[x]["num"]}')
                # 重置大循环记录
                ref_idx += 1    # 当前大循环数
                cycle_record = {x[0]:{'num':0,
                                        # 'value':{attribute:[] for attribute in attributes}
                                    }for x in np.unique(data["data"]["step"][0, 0]['comment'])}
        # 输出全部信息
        print('\n'+'='*20+f' overall '+'='*20)
        for x in record:
            print(f'{x}:{record[x]}')


    #%% 有效数据提取

    # 定义记录的类
    class RW_cycle:
        def __init__(self,idx):
            self.idx = idx
            self.charge = None
            self.discharge= []
            self.merged_discharge= None
            self.capacity = 0
 
        def __repr__(self):
            return f"No.{self.idx} {0 if self.charge is None else 1}C {self.discharge.__len__()}D"

    class Data:
        def __init__(self):
            self.all_useful=[]
            self.merged_all_useful=[]
            self.RW_cycles = [RW_cycle(idx) for idx in range(0,200)]
            self.cur = 0
        
        @property
        def cur_cycle(self):
            return self.RW_cycles[self.cur]
        
        @property
        def num_RW_cycles(self):
            return self.RW_cycles.__len__()

        @property
        def num_useful_info(self):
            return self.all_useful.__len__()

        def next_RW_cycle(self):
            self.cur += 1
        
        def __repr__(self):
            return  f'{self.num_useful_info} useful infos'

    class Cycle_record:
        def __init__(self,a):
            self.ref_idx = 0
            self.num = 0
            self.capacity = 0
            self.start_time = 0
            self.data = Data()
        def __repr__(self):
            return f'{self.num} infos, capacity={self.capacity:.3f}'

    # 记录的实例
    cycle_records = [Cycle_record(a) for a in range(50)]
    ref_idx = 0
    record = {x[0]:0 for x in np.unique(data["data"]["step"][0, 0]['comment'])} # 全局统计

    # 对每一条操作进行遍历
    for i in range(data["data"]["step"][0, 0].shape[1]):

        # 当前操作信息处理
        cur = data["data"]["step"][0, 0]['comment'][0,i][0]         # 是操作的类型，如'rest (random walk)'
        cycle_records[ref_idx].num += 1
        record[cur]+=1

        # 如果是有用的操作，则记录下这条操作的所有信息
        if cur in useful_comments:
            cur_data = np.concatenate(
                        tuple(data["data"]["step"][0,0][attribute][0,i] for attribute in attributes)
                        )
            if cur == 'charge (after random walk discharge)':
                cycle_records[ref_idx].data.cur_cycle.charge = cur_data
                cycle_records[ref_idx].data.next_RW_cycle()
            if cur == 'discharge (random walk)':
                cycle_records[ref_idx].data.cur_cycle.discharge.append(cur_data)
            cycle_records[ref_idx].data.all_useful.append(cur_data)

        # 如果是'reference discharge'，则算出capacity
        if cur == 'reference discharge':
            capacity = np.trapz(
                                data["data"]["step"][0,0]['current']     [0,i][0,:],
                                data["data"]["step"][0,0]['relativeTime'][0,i][0,:]/3600,
                                )
            cycle_records[ref_idx].capacity = capacity
            cycle_records[ref_idx].start_time = data["data"]["step"][0,0]['time'][0,i][0,0]

            print(f'reference discharge at {i} capacity is {capacity:.5f}')

        # 如果出现 'rest post pulsed load'（大循环结束标志），则处理上一个大循环的信息，并重置大循环记录 
        if cur == 'rest post pulsed load':
            # 打印当前大循环记录的信息
            cycle_records[ref_idx].ref_idx = ref_idx
            print('='*20+f" cycle {cycle_records[ref_idx].ref_idx} "+'='*20)
            print(f"num:      {cycle_records[ref_idx].num}")
            print(f"capacity: {cycle_records[ref_idx].capacity:.5f}")
            print(f"data:     {cycle_records[ref_idx].data}")
            # 重置大循环记录
            ref_idx += 1    # 当前大循环数

    # 输出全局统计
    print('\n'+'='*20+f' overall '+'='*20)
    for x in record:
        print(f'{x}:{record[x]}')



    #%% 数据合并处理
    # 缺失值删除
    for ref_idx in range(len(cycle_records)-1,-1,-1):
        if cycle_records[ref_idx].num == 0:
            del cycle_records[ref_idx]
            continue
        for idx in range(cycle_records[ref_idx].data.num_RW_cycles-1,-1,-1):
            if cycle_records[ref_idx].data.RW_cycles[idx].discharge.__len__() == 0:
                del cycle_records[ref_idx].data.RW_cycles[idx]
                continue
            elif cycle_records[ref_idx].data.RW_cycles[idx].charge is None:
                del cycle_records[ref_idx].data.RW_cycles[idx]
                continue


    # all合并（时间间隔一样）
    for ref_idx in range(len(cycle_records)):
        try:
            cycle_records[ref_idx].data.merged_all_useful = np.concatenate(
                tuple(cycle_records[ref_idx].data.all_useful[i] for i in range(cycle_records[ref_idx].data.num_useful_info)),axis=1
            )
        except Exception as e:
            print(f"{e} at ref_idx:{ref_idx}")

    # 合并一个RWcycle中的discharge
    for ref_idx in range(len(cycle_records)):
        for idx in range(cycle_records[ref_idx].data.num_RW_cycles):
            try:
                cycle_records[ref_idx].data.RW_cycles[idx].merged_discharge = np.concatenate(
                    tuple(cycle_records[ref_idx].data.RW_cycles[idx].discharge[i] 
                        for i in range(cycle_records[ref_idx].data.RW_cycles[idx].discharge.__len__())),axis=1
                )
            except Exception as e:
                print(f"{e} at ref_idx:{ref_idx},idx:{idx}")



    #%% 填补SOH空缺值（插值）
    # ref中测定的capacity和对应时间
    t_p=[]
    capacity_p=[]
    for i in range(len(cycle_records)):
        if cycle_records[i].capacity !=  0:
            t_p.append(cycle_records[i].start_time)
            capacity_p.append(cycle_records[i].capacity)
    init_capacity = capacity_p[0]

    # 需要插值的点，即RW_cycle对应的时间，以放电结束充电开始的时间点为准
    t = []
    for ref_idx in range(len(cycle_records)):
        for idx in range(cycle_records[ref_idx].data.num_RW_cycles):
            t.append(
                cycle_records[ref_idx].data.RW_cycles[idx].charge[0,0]
            )

    # 插值
    # f = interpolate.interp1d(t_p,capacity_p,kind='quadratic')
    f = interpolate.CubicSpline(t_p,capacity_p,bc_type="natural",extrapolate=True)
    capacity = f(t)

    # # 显示结果
    # plt.plot(t,capacity,'r')
    # plt.plot(t_p,capacity_p,'b')
    # plt.show()

    # 把对应的capacity放对应RWcycle
    counter = 0
    for ref_idx in range(len(cycle_records)):
        for idx in range(cycle_records[ref_idx].data.num_RW_cycles):
            cycle_records[ref_idx].data.RW_cycles[idx].capacity = capacity[counter]
            counter += 1

    # 检测放回结果
    temp_t = []
    temp_c = []
    for ref_idx in range(len(cycle_records)):
        for idx in range(cycle_records[ref_idx].data.num_RW_cycles):
            temp_c.append(cycle_records[ref_idx].data.RW_cycles[idx].capacity)
            temp_t.append(cycle_records[ref_idx].data.RW_cycles[idx].charge[0,0])
    # plt.plot(t_p,capacity_p,'b')
    # plt.plot(t,capacity,'r')
    # plt.plot(temp_t,temp_c,'g')
    # plt.show()

    #%% 整合所有的RWcycle数据
    all_rw_cycles = []
    for ref_idx in range(len(cycle_records)):
        for idx in range(cycle_records[ref_idx].data.num_RW_cycles):
            all_rw_cycles.append(cycle_records[ref_idx].data.RW_cycles[idx])

    # 输出cycle长度等数据
    charge_length = []
    discharge_length = []
    for rw_cycle in all_rw_cycles:
        charge_length.append(rw_cycle.charge.shape[1])
        discharge_length.append(rw_cycle.merged_discharge.shape[1])
    
    # plt.hist(charge_length)
    # plt.show()
    # plt.hist(discharge_length)
    # plt.show()


    #%% 添加随机噪声，偏移
    def add_noise(array,noise_percent=0.02):
        time = array[0:2,:]
        result = np.zeros_like(array,dtype='float64')
        for i in range(2,array.shape[0]):
            mean = array[i,:].mean()
            noise = np.random.randn(array.shape[1])*mean*noise_percent
            noise = noise - np.mean(noise)
            result[i,:] = array[i,:]+noise
        result[0:2,:] = time
        return result

    def add_bias(array,
                voltage_bias,
                current_bias,
                temperature_bias):
        time = array[0:2,:]
        result = np.zeros_like(array,dtype='float64')
        result[2,:] = array[2,:] + voltage_bias
        result[3,:] = array[3,:] + current_bias
        result[4,:] = array[4,:] + temperature_bias
        result[0:2,:] = time
        return result

    print(f"{len(all_rw_cycles)} all_rw_cycles")

    # expand_multiple = 1    # 扩展的倍数
    new_rw_cycles = []
    for rw_cycle in all_rw_cycles:
        for i in range(expand_multiple-1):
            new_rw_cycle = RW_cycle(rw_cycle.idx+1000*i)
            new_rw_cycle.capacity = rw_cycle.capacity

            noise_percent =     (random.random()-0.5)*0.02      # 白噪声强度
            voltage_bias =      (random.random()-0.5)*0.1       # 各项的偏移量
            current_bias =      (random.random()-0.5)*0.005     # 各项的偏移量
            temperature_bias =  (random.random()-0.5)*2.5       # 各项的偏移量
            new_charge = add_noise(rw_cycle.charge,noise_percent)
            new_charge = add_bias( new_charge,voltage_bias,current_bias,temperature_bias)
            new_rw_cycle.charge = new_charge
            try:
                new_merged_discharge = add_noise(rw_cycle.merged_discharge,noise_percent)
                new_merged_discharge = add_bias( new_merged_discharge,voltage_bias,current_bias,temperature_bias)
                new_rw_cycle.merged_discharge = new_merged_discharge
            except Exception as e:
                print(f"{e} at RW_cycle {rw_cycle.__repr__()}")
                break
            new_rw_cycles.append(new_rw_cycle)
    all_rw_cycles = all_rw_cycles+new_rw_cycles

    print(f"{len(all_rw_cycles)} all_rw_cycles after expand")

    #%% 分类输出
    class SingleRecord:
        def __init__(self,type):
            self.SOH = 0
            self.data = None
            self.type = type
        
        def __repr__(self):
            return f"shape {self.data.shape} type {self.type}"

    all_charge_records = []
    all_discharge_records = []
    for rw_cycle in all_rw_cycles:
        # charge_record = SingleRecord('C')
        # charge_record.data = rw_cycle.charge[2:,:]
        # charge_record.SOH = rw_cycle.capacity / init_capacity
        # all_charge_records.append(charge_record)

        if "charge" in output_type or "C" in output_type: 
            for i in range(rw_cycle.charge.shape[1]//squence_length):
                charge_record = SingleRecord('C')
                charge_record.data = rw_cycle.charge[2:,i*squence_length:(i+1)*squence_length]
                charge_record.SOH = rw_cycle.capacity / init_capacity
                all_charge_records.append(charge_record)
            if rw_cycle.charge.shape[1] - squence_length > 0 and rw_cycle.charge.shape[1] % squence_length > squence_length*0.5:
                charge_record = SingleRecord('C')
                charge_record.data = rw_cycle.charge[2:,-squence_length:]
                charge_record.SOH = rw_cycle.capacity / init_capacity
                all_charge_records.append(charge_record)

        if "discharge" in output_type or "D" in output_type:
            for i in range(rw_cycle.merged_discharge.shape[1]//squence_length):
                discharge_record = SingleRecord('D')
                discharge_record.data = rw_cycle.merged_discharge[2:,i*squence_length:(i+1)*squence_length]
                discharge_record.SOH = rw_cycle.capacity / init_capacity
                all_discharge_records.append(discharge_record)
            if rw_cycle.merged_discharge.shape[1] - squence_length > 0 and rw_cycle.merged_discharge.shape[1] % squence_length > squence_length*0.5:
                discharge_record = SingleRecord('D')
                discharge_record.data = rw_cycle.merged_discharge[2:,-squence_length:]
                discharge_record.SOH = rw_cycle.capacity / init_capacity
                all_discharge_records.append(discharge_record)


    return all_charge_records,all_discharge_records,charge_length,discharge_length
    #%% 
#%%
if __name__ == '__main__':
    getdata(squence_length = 150,expand_multiple = 10,output_type=["C","D"])