# 主要改动为在pid_cart和pid_pole加入注释(即我自己的理解)

import random
import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import *
from matplotlib import animation

from cartpole_env import *
from matplotlib.pylab import mpl
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib import font_manager
font_path = "./AiDianFengYaHei（ShangYongMianFei）-2.ttf"  # 改成正确路径
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

kp_cart = 2
kd_cart = 50
kp_pole = 8
kd_pole = 100
DIRECT_MAG=True
RANDOM_NOISE=False



if DIRECT_MAG:
    env=CartPoleEnv()
else:
    env = gym.make('CartPole-v1')

class CartPoleControl:

    def __init__(self, kp_cart, kd_cart, kp_pole, kd_pole):
        self.kp_cart = kp_cart
        self.kd_cart = kd_cart
        self.kp_pole = kp_pole
        self.kd_pole = kd_pole
        self.bias_cart_1 = 0
        self.bias_pole_1 = 0
        self.i=0

    def pid_cart(self, position):
        ''' 某个类中的pid部分代码
            小车需要时刻待在屏幕正中间
        Input:
            position : 当前时刻小车的位置

        Return:
            balance  : 需要小车回正的"力气"
        
        Parameters:
            bias     : 小车的位置偏差
            d_bias   : 小车的速度偏差
            kp_cart  : kp P系数
            kd_cart  : kd D系数
            bias_cart_1 : 上一次的位置偏差
        '''

        # 计算偏差
        bias = position

        # 计算速度
        d_bias = bias - self.bias_cart_1

        # 核心公式
        balance = self.kp_cart * bias + self.kd_cart * d_bias
        
        # 保存记忆
        self.bias_cart_1 = bias
        return balance

    def pid_pole(self, angle):
        '''控制杆子平衡
        
        Input:
            angle:杆子当前的角度
        
        Return:
            balance:最终输出的力气
        
        Parameters:
            bias     : 杆子的角度偏差
            d_bias   : 杆子的速度偏差
            kp_pole  : kp P系数
            kd_pole  : kd D系数
            bias_pole_1 : 上一次的位置偏差
        '''

        # 计算角度偏差
        bias = angle  

        # 计算倒下的速度
        d_bias = bias - self.bias_pole_1

        # 核心公式
        balance = -self.kp_pole * bias - self.kd_pole * d_bias
        
        # 保存记忆
        self.bias_pole_1 = bias
        return balance

    def control_output(self, control_cart, control_pole):
        if DIRECT_MAG:
            return -10*(control_pole - control_cart)
        else:
            return 1 if (control_pole - control_cart) < 0 else 0

def save_frames_as_gif(frames, path):
    filename = 'PID_'+ 'CartPole-v1' + '.gif'
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='pillow')

if __name__ == '__main__':

    control=CartPoleControl(kp_cart, kd_cart, kp_pole, kd_pole)

    rewards=0
    state = env.reset()
    k_steps = 401
    X_k = np.zeros((4, k_steps+1))
    X_k[:,0] = state
    # 开辟所有控制输入u的存储空间
    U_k = np.zeros((1, k_steps))
    done = False
    i=0
    step_a = 0
    frames = []
    while abs(state[2]<2) & (step_a < k_steps):
        env.render()
        frames.append(env.render(mode='rgb_array'))
        control_pole = control.pid_pole(state[2])
        control_cart = control.pid_cart(state[0])
        if RANDOM_NOISE and random.random()>0.99:
            i=2

        if i>0:
            if DIRECT_MAG:
                action = 10
            else:
                action=1
            i-=1
        else:
            action = control.control_output(control_cart, control_pole)

        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards+=reward
        X_k[:,step_a+1] = state
        U_k[:,step_a] = action
        step_a += 1
        print(step_a)
    env.close()
    save_frames_as_gif(frames, path = './')


    # 绘制结果
    plt.subplot(2, 1, 1)
    plt.plot(X_k[0, :], label=f"x")
    plt.plot(X_k[1, :], label=f"x_dot")
    plt.plot(X_k[2, :], label=f"theta")
    plt.plot(X_k[3, :], label=f"theta_dot")
    plt.legend()
    plt.title("PID状态变量")
    plt.xlabel("时间步")
    plt.ylabel("状态值")
    
    # 第二个子图: 控制输入
    plt.subplot(2, 1, 2)
    for i in range(U_k.shape[0]):
        plt.plot(U_k[i, :], label=f"u{i+1}")
    plt.legend()
    plt.title("PID控制输入")
    plt.xlabel("时间步")
    plt.ylabel("控制输入值")
    
    # 调整布局并显示
    plt.tight_layout()
    plt.savefig('PID', dpi=1000)
    plt.show()