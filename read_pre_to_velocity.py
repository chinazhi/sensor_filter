import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from filter import OneDimKalmanFilter

# 文件路径
pre_file_path = './raw_data/p1-13.txt'

# 初始化列表保存时间和z值
pre_times = []
pre_values = []
pre_values_av = []
# 读取文件并提取数据
with open(pre_file_path, 'r') as file:
    for line in file:
        # 匹配时间和z值的正则表达式
        match = re.match(r'\[(.*?)\]\[__main__\]\[debug\] pressure (-?\d+) average pressure (-?\d+)', line)
        if match:
            time_str = match.group(1)
            p1_value = int(match.group(2))
            p2_value = int(match.group(3))
            # 解析时间字符串为datetime对象
            time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S:%f')
            pre_times.append(time_obj)
            pre_values.append(p1_value)
            pre_values_av.append(p2_value)

# 零点校正：计算静止时的平均加速度值并减去该偏置
static_pressure_mean = np.mean(pre_values[:20])  # 假设前20个值是静止时的
print('Static pressure Mean:', static_pressure_mean)
pre_values_corrected = pre_values - static_pressure_mean

# 初始参数设置
initial_process_variance = 0.0062
initial_measurement_variance = 1.2
initial_estimated_error = 1
initial_value = pre_values_corrected[0]

# 初始化卡尔曼滤波器并绘制滤波后的数据
kf = OneDimKalmanFilter(initial_process_variance, initial_measurement_variance, initial_estimated_error, initial_value)
filtered_acceleration = [kf.update(z) for z in pre_values_corrected]

# 创建列表存储时间和气压
time_list = pre_times
pressure_list = filtered_acceleration

# 创建DataFrame
df = pd.DataFrame({'time': time_list, 'pressure': pressure_list})

# 已知1楼和顶楼的气压及高度差
P1 = 101228
P2 = 100820
H = 40
P_avg = (P1 + P2) / 2

# 计算校准常数
k = (P1 - P2) / (H * P_avg)

# 计算时间差（秒）
df['time_diff'] = df['time'].diff().dt.total_seconds()

# 计算高度差 (近似公式)
df['height_diff'] = df['pressure'].diff() / (df['pressure'] * k)  # 这里的 0.00012 是一个近似值，可以根据具体情况调整

# 计算速度（高度差 / 时间差）
df['velocity'] = df['height_diff'] / df['time_diff']

# 确定运行方向
df['direction'] = df['height_diff'].apply(lambda x: 'up' if x > 0 else 'down')

# 打印结果
# print(df[['time', 'pressure', 'height_diff', 'velocity', 'direction']])

# 绘制速度图
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['velocity'], label='Velocity (m/s)')
plt.xlabel('Time')
plt.ylabel('Velocity (m/s)')
plt.title('Elevator Speed vs Time')
plt.legend()
plt.show()
