class OneDimKalmanFilter:
    """
    初始化一维卡尔曼滤波器对象

    参数:
    - process_variance: 过程噪声方差，表示系统模型中内在的噪声方差，用于描述模型估计状态的不确定性。
    - measurement_variance: 测量噪声方差，表示传感器测量的噪声方差，用于描述测量数据的不确定性。
    - estimated_error: 初始估计误差，表示初始状态估计的不确定性。
    - initial_value: 初始值，表示初始状态的估计值。
    """
    def __init__(self, process_variance, measurement_variance, estimated_error, initial_value):
        self.process_variance = process_variance            # 过程噪声方差
        self.measurement_variance = measurement_variance    # 测量噪声方差
        self.posteri_error = estimated_error                # 后验估计误差
        self.posteri_estimate = initial_value               # 后验估计值

    def update(self, measurement):
        """
        更新一维卡尔曼滤波器的状态估计

        参数:
        - measurement: 当前时刻的测量值

        返回值:
        - posteriori_estimate: 更新后的后验估计值，即当前时刻的状态估计值
        """
        priori_estimate = self.posteri_estimate     # 先验估计值为上一时刻的后验估计值
        priori_error = self.posteri_error + self.process_variance # 先验估计误差为上一时刻的后验误差加上过程方差

        blending_factor = priori_error / (priori_error + self.measurement_variance) # 计算融合因子
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate) # 更新后验估计值
        self.posteri_error = (1 - blending_factor) * priori_error   # 更新后验估计误差

        return self.posteri_estimate     # 返回更新后的后验估计值
