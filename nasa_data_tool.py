# coding:UTF-8
def get_data(path,verbose=False):
    #%% 导入模块
    from numpy.lib.function_base import average
    import scipy.io as scio
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import interpolate
    import random

    # dataFile = r"./dataset/nasa/RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22.mat"
    dataFile = path
    data = scio.loadmat(dataFile)

    useful_comments = [
        'charge (after random walk discharge)',
        'discharge (random walk)',
    ]

    attributes = [  
    'time',
    'relativeTime',
    'voltage',
    'current',
    'temperature',
    ]


    #%% 有效数据提取

    # 定义记录的类
    class RW_cycle:
        def __init__(self,idx):
            self.idx = idx
            self.charge = None
            self.discharge= []
            self.merged_discharge= None
            self.capacity = 0

        @property
        def num_discharges(self):
            return len(self.discharge)
        
        def __repr__(self):
            return f"No.{self.idx} {0 if self.charge is None else 1}C {self.discharge.__len__()}D"

    class Data:
        def __init__(self):
            self.all_useful=[]
            self.merged_all_useful=[]
            self.RW_cycles = [RW_cycle(idx) for idx in range(0,52)]
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
    cycle_records = [Cycle_record(a) for a in range(13)]
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
            if verbose:
                print(f'reference discharge at {i} capacity is {capacity:.5f}')

        # 如果出现 'rest post pulsed load'（大循环结束标志），则处理上一个大循环的信息，并重置大循环记录 
        if cur == 'rest post pulsed load':
            # 打印当前大循环记录的信息
            cycle_records[ref_idx].ref_idx = ref_idx
            if verbose:
                print('='*20+f" cycle {cycle_records[ref_idx].ref_idx} "+'='*20)
                print(f"num:      {cycle_records[ref_idx].num}")
                print(f"capacity: {cycle_records[ref_idx].capacity:.5f}")
                print(f"data:     {cycle_records[ref_idx].data}")
            # 重置大循环记录
            ref_idx += 1    # 当前大循环数
    if verbose:
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
            if cycle_records[ref_idx].data.RW_cycles[idx].charge is None:
                del cycle_records[ref_idx].data.RW_cycles[idx]
                continue

    # 合并一个RWcycle中的discharge
    for ref_idx in range(len(cycle_records)):
        for idx in range(cycle_records[ref_idx].data.num_RW_cycles):
            try:
                cycle_records[ref_idx].data.RW_cycles[idx].merged_discharge = np.concatenate(
                    tuple(cycle_records[ref_idx].data.RW_cycles[idx].discharge[i] for i in range(cycle_records[ref_idx].data.RW_cycles[idx].num_discharges)),axis=1
                )
            except Exception as e:
                if verbose:
                    print(f"{e} at ref_idx:{ref_idx},idx:{idx}")

    # all合并（时间间隔一样，插值）
    for ref_idx in range(len(cycle_records)):
        try:
            cycle_records[ref_idx].data.merged_all_useful = np.concatenate(
                tuple(cycle_records[ref_idx].data.all_useful[i] for i in range(cycle_records[ref_idx].data.num_useful_info)),axis=1
            )
        except Exception as e:
            if verbose:
                print(f"{e} at ref_idx:{ref_idx}")


    #%% 填补SOH空缺值（插值）
    # ref中测定的capacity和对应时间
    t_p=[]
    capacity_p=[]
    for i in range(len(cycle_records)):
        if cycle_records[i].capacity is not 0:
            t_p.append(cycle_records[i].start_time)
            capacity_p.append(cycle_records[i].capacity)

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

    #%% 整合所有的RWcycle数据
    all_rw_cycles = []
    for ref_idx in range(len(cycle_records)):
        for idx in range(cycle_records[ref_idx].data.num_RW_cycles):
            all_rw_cycles.append(cycle_records[ref_idx].data.RW_cycles[idx])

    #%% 添加随机噪声，偏移
    def add_noise(array,noise_percent=0.02):
        time = array[0:2,:]
        result = np.zeros_like(array,dtype='float64')
        for i in range(array.shape[0]):
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

    if verbose:
        print(f"{len(all_rw_cycles)} all_rw_cycles")

    expand_multiple = 60    # 扩展的倍数
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
                if verbose:
                    print(f"{e} at RW_cycle {rw_cycle.__repr__()}")
                break
            new_rw_cycles.append(new_rw_cycle)
    all_rw_cycles = all_rw_cycles+new_rw_cycles
    if verbose:
        print(f"{len(all_rw_cycles)} all_rw_cycles after expand")

    return all_rw_cycles