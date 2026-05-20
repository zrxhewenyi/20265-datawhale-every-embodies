class PID:
    # pid的初始化赋值
    def __init__(self, Kp, Ki, Kd, setpoint=0, sample_time=0.01):
        '''
        Input:
            Kp:比例系数
            Ki:比例系数
            Kd:比例系数
            setpoint:比例系数
            sample_time:采样时间
            prev_error:上一帧的误差
            integral:历史误差的总和
        '''
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd 
        self.setpoint = setpoint
        self.sample_time = sample_time
 
        self.prev_error = 0
        self.integral = 0

    # pid的cal_process
    def update(self, measured_value):
        '''计算函数

        Input:
            measured_value:传感器的真实反馈

        Return:
            output:最终输出的指令

        Parameters:
            self.__ : 具体见__init__函数
            error:最终输出的指令
            derivative:误差变化的速度
        '''
        error = self.setpoint - measured_value # 计算误差P
        self.integral += error * self.sample_time # 积分I
        derivative = (error - self.prev_error) / self. sample_time # 微分D
 
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative # 计算控制输入
        self.prev_error = error # 保存误差
 
        return output
 

pid = PID(Kp=1.0, Ki=0.1, Kd=0.01, setpoint=100)
measured_value = 90  # 假设的当前测量值
control_input = pid.update(measured_value)
 
print(f"Control Input: {control_input}")