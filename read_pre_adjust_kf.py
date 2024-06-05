import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.widgets import Slider
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

# 初始参数设置
initial_process_variance = 0.0062
initial_measurement_variance = 1.2
initial_estimated_error = 1
initial_value = pre_values_corrected[0]

# 初始化卡尔曼滤波器并绘制滤波后的数据
kf = OneDimKalmanFilter(initial_process_variance, initial_measurement_variance, initial_estimated_error, initial_value)
filtered_acceleration = [kf.update(z) for z in pre_values_corrected]

# 创建图形和滑动条
fig, ax = plt.subplots(figsize=(15, 8))
plt.subplots_adjust(left=0.1, bottom=0.25)

# 绘制原始数据
line_original, = ax.plot(pre_times, pre_values_corrected, 'b-', marker='o', alpha=0.5, label='Original Z Values')
# 绘制滤波后的数据
line_filtered, = ax.plot(pre_times, filtered_acceleration, 'r-', marker='o', label='Filtered Z Values')

ax.set_xlabel('Time')
ax.set_ylabel('pressure')
ax.set_title('pressure vs. Time')
ax.legend()
plt.xticks(rotation=45)
plt.grid(True)

# 添加滑动条
axcolor = 'lightgoldenrodyellow'
ax_process_variance = plt.axes([0.1, 0.08, 0.65, 0.03], facecolor=axcolor)
ax_measurement_variance = plt.axes([0.1, 0.03, 0.65, 0.03], facecolor=axcolor)

s_process_variance = Slider(ax_process_variance, 'Process Var', 1e-6, 1e-0, valinit=initial_process_variance, valstep=1e-6)
s_measurement_variance = Slider(ax_measurement_variance, 'Measurement Var', 0.1, 30.0, valinit=initial_measurement_variance, valstep=0.1)

# 更新函数
def update(val):
    process_variance = s_process_variance.val
    measurement_variance = s_measurement_variance.val

    kf = OneDimKalmanFilter(process_variance, measurement_variance, initial_estimated_error, initial_value)
    filtered_acceleration = [kf.update(z) for z in pre_values_corrected]
    line_filtered.set_ydata(filtered_acceleration)
    fig.canvas.draw_idle()

# 注册滑动条的更新函数
s_process_variance.on_changed(update)
s_measurement_variance.on_changed(update)

# 显示图形
plt.show()

