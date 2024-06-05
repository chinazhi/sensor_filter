import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.widgets import Slider
from filter import OneDimKalmanFilter

# 计算速度（加速度的积分）
def calculate_velocity(acceleration, times):
    velocity = [0]  # 初始速度假设为0
    for i in range(1, len(acceleration)):
        dt = (times[i] - times[i - 1]).total_seconds()  # 时间差，以秒为单位
        velocity.append(velocity[-1] + acceleration[i] * dt)
    return velocity

# 读取文件并提取数据
file_path = './raw_data/1-9.txt'
times = []
z_values = []

with open(file_path, 'r') as file:
    for line in file:
        match = re.match(r'\[(.*?)\]\[__main__\]\[debug\] z (-?\d+) average y data (-?\d+)', line)
        if match:
            time_str = match.group(1)
            z_value = int(match.group(2))
            time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S:%f')
            times.append(time_obj)
            z_values.append(z_value)

# 零点校正
static_acceleration_mean = np.mean(z_values[:-10])
z_values_corrected = np.array(z_values) - static_acceleration_mean

# 初始参数设置
initial_process_variance = 0.177
initial_measurement_variance = 4.2
initial_estimated_error = 1
initial_value = z_values_corrected[0]

# 初始化卡尔曼滤波器并绘制滤波后的数据
kf = OneDimKalmanFilter(initial_process_variance, initial_measurement_variance, initial_estimated_error, initial_value)
filtered_acceleration = [kf.update(z) for z in z_values_corrected]

velocity_kf = calculate_velocity(filtered_acceleration, times)

# 创建图形和滑动条
fig, ax = plt.subplots(figsize=(15, 8))
plt.subplots_adjust(left=0.1, bottom=0.25)

# 绘制原始数据
line_original, = ax.plot(times, z_values_corrected, 'b-', marker='o', alpha=0.5, label='Original Z Values')
# 绘制滤波后的数据
line_filtered, = ax.plot(times, filtered_acceleration, 'r-', marker='o', label='Filtered Z Values')
# 绘制速度
line_velocity, = ax.plot(times, velocity_kf, 'g-', marker='o', label='Velocity')

or_velocity_kf = calculate_velocity(z_values_corrected, times)
line_or_velocity, = ax.plot(times, or_velocity_kf, 'k-', marker='o', alpha=0.5, label='Original or Velocity')

ax.set_xlabel('Time')
ax.set_ylabel('Acceleration (z)')
ax.set_title('Acceleration vs. Time')
ax.legend()
plt.xticks(rotation=45)
plt.grid(True)

# 添加滑动条
axcolor = 'lightgoldenrodyellow'
ax_process_variance = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_measurement_variance = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)

s_process_variance = Slider(ax_process_variance, 'Process Var', 1e-6, 1e-0, valinit=initial_process_variance, valstep=1e-6)
s_measurement_variance = Slider(ax_measurement_variance, 'Measurement Var', 0.1, 30.0, valinit=initial_measurement_variance, valstep=0.1)

# 更新函数
def update(val):
    process_variance = s_process_variance.val
    measurement_variance = s_measurement_variance.val

    kf = OneDimKalmanFilter(process_variance, measurement_variance, initial_estimated_error, initial_value)
    filtered_acceleration = [kf.update(z) for z in z_values_corrected]
    velocity_kf = calculate_velocity(filtered_acceleration, times)
    line_filtered.set_ydata(filtered_acceleration)
    line_velocity.set_ydata(velocity_kf)
    fig.canvas.draw_idle()

# 注册滑动条的更新函数
s_process_variance.on_changed(update)
s_measurement_variance.on_changed(update)

# 显示图形
plt.show()

