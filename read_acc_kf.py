import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from filter import OneDimKalmanFilter

# 定义移动平均滤波器
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# 计算速度（加速度的积分）
def calculate_velocity(acceleration, times):
    velocity = [0]  # 初始速度假设为0
    for i in range(1, len(acceleration)):
        dt = (times[i] - times[i - 1]).total_seconds()  # 时间差，以秒为单位
        velocity.append(velocity[-1] + acceleration[i] * dt)
    return velocity

# 文件路径
acc_file_path = './raw_data/7-1.txt'

# 初始化列表保存时间和z值
times = []
z_values = []
z_values_av = []
# 读取文件并提取数据
with open(acc_file_path, 'r') as file:
    for line in file:
        # 匹配时间和z值的正则表达式
        match = re.match(r'\[(.*?)\]\[__main__\]\[debug\] z (-?\d+) average y data (-?\d+)', line)
        if match:
            time_str = match.group(1)
            z_value = int(match.group(2))
            y_value = int(match.group(3))
            # 解析时间字符串为datetime对象
            time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S:%f')
            times.append(time_obj)
            z_values.append(z_value)
            z_values_av.append(y_value)


# 零点校正：计算静止时的平均加速度值并减去该偏置
static_acceleration_mean = np.mean(z_values[:-10])  # 假设前20个值是静止时的
print('Static Acceleration Mean:', static_acceleration_mean)
z_values_corrected = z_values - static_acceleration_mean


# 应用移动平均滤波器
window_size = 5
z_values_ma = moving_average(z_values_corrected, window_size)

# 为了对齐移动平均数据的时间轴，需要调整时间列表
times_ma = times[(window_size-1)//2: -(window_size//2)]

velocity_z_ave = calculate_velocity(z_values_ma, times_ma)


# 定义卡尔曼滤波器参数
process_variance = 0.6882  # 过程噪声
measurement_variance = 30  # 测量噪声
estimated_error = 1  # 初始估计误差
initial_value = z_values_corrected[0]  # 初始值

# 创建卡尔曼滤波器实例
kf = OneDimKalmanFilter(process_variance, measurement_variance, estimated_error, initial_value)

# 使用卡尔曼滤波器平滑加速度数据
z_values_kf = []
for measurement in z_values_corrected:
    z_values_kf.append(kf.update(measurement))

# 加速度数据计算速度
velocity_kf = calculate_velocity(z_values_kf, times)

# 绘制折线图 设置图的大小为10x6
plt.figure(figsize=(15, 8))

# 绘制加速度数据
plt.subplot(2, 1, 1)
plt.plot(times, z_values_corrected, marker='o', linestyle='-', color='g', alpha=0.5, label='Corrected Z Values')
plt.plot(times_ma, z_values_ma, marker='o', linestyle='-', color='r', alpha=0.5, label='Average Z Values')
plt.plot(times, z_values_kf, marker='o', linestyle='-', color='b', alpha=0.5, label='Kalman Z Values')
plt.xlabel('time')
plt.ylabel('Acceleration (z)')
plt.title('Acceleration vs. Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# 绘制速度数据
plt.subplot(2, 1, 2)
plt.plot(times_ma, velocity_z_ave, marker='o', linestyle='-', color='r', label='Velocity average Z')
plt.plot(times, velocity_kf, marker='o', linestyle='-', color='b', label='Velocity kalman Z')
plt.xlabel('time')
plt.ylabel('Velocity')
plt.title('Velocity vs. Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# 调整布局使得子图适应画布
plt.tight_layout()
# 展示图形
plt.show()
