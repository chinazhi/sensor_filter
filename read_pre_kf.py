import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from filter import OneDimKalmanFilter

# 定义移动平均滤波器
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

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

# 应用移动平均滤波器
window_size = 5
pre_values_ma = moving_average(pre_values_corrected, window_size)

# 为了对齐移动平均数据的时间轴，需要调整时间列表
times_ma = pre_times[(window_size-1)//2: -(window_size//2)]


# 定义卡尔曼滤波器参数
process_variance = 1e-3  # 过程噪声
measurement_variance = 0.1  # 测量噪声
estimated_error = 1  # 初始估计误差
initial_value = pre_values_corrected[0]  # 初始值

# 创建卡尔曼滤波器实例
kf = OneDimKalmanFilter(process_variance, measurement_variance, estimated_error, initial_value)

# 使用卡尔曼滤波器平滑加速度数据
pre_values_kf = [kf.update(measurement) for measurement in pre_values_corrected]

# 绘制折线图 设置图的大小为10x6
plt.figure(figsize=(15, 8))

# 绘制加速度数据
plt.subplot(2, 1, 1)

plt.plot(pre_times, pre_values_corrected, marker='o', linestyle='-', color='r', alpha=0.5, label='original pressure Values')
plt.plot(times_ma, pre_values_ma, marker='o', linestyle='-', color='g', alpha=0.5, label='average pressure Values')
plt.plot(pre_times, pre_values_kf, marker='o', linestyle='-', color='b', alpha=0.5, label='Kalman pressure Values')
plt.xlabel('time')
plt.ylabel('Acceleration (z)')
plt.title('Acceleration vs. Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# # 绘制速度数据
# plt.subplot(2, 1, 2)
# plt.plot(times_ma, velocity_z_ave, marker='o', linestyle='-', color='r', label='Velocity average Z')
# plt.plot(times, velocity_kf, marker='o', linestyle='-', color='b', label='Velocity kalman Z')
# plt.xlabel('time')
# plt.ylabel('Velocity')
# plt.title('Velocity vs. Time')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.legend()

# 调整布局使得子图适应画布
plt.tight_layout()
# 展示图形
plt.show()
