import numpy as np

class RT1ActionTokenzizer:
    def __init__(self,action_min=-1.0, action_max=1.0, vocab_size=256):
        # 初始化函数
        """
        Input:
            action_min, action_max:
                # 机械臂在物理世界中运动时，它的各项控制量都是有极限的，不可能无限快，也不可能无限大
                # 例如：如果动作代表平移速度：机械臂前后移动的最大速度可能是向后 1.0 m/到向前 1.0 m/s
                为了防止尺度失衡，算法通常会提前把所有复杂的关节、速度指令统一缩放到[−1.0,1.0]之间
            vocab_size:
                # 将连续数据切割成了多少份
                离散格子的分辨率与控制精度
        """
        self.min = action_min
        self.max = action_max
        self.vocab_size = vocab_size

    def tokenize(self, action_continuous):
        # 将连续动作[-1, 1]转换为离散Token[0, 255]
        '''
        Input:
            action_continuous:连续空间的原始动作值            
                
        Return:
            action_tokens:有限的整数分类号
        '''

        # 1.归一化到[0, 1]
        action_norm = (action_continuous - self.min) / (self.max - self.min) # Min-Max Normalization
        action_norm = np.clip(action_norm, 0, 1)

        '''
        clip:裁剪函数，强行将数据划归到原来的范围中去
            a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            np.clip(a, 1, 8)
            np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        '''


        # 2.映射到[0, vocal_size - 1]
        action_tokens = np.floor(action_norm * (self.vocab_size - 1)).astype(np.int32)

        '''
        floor:向下取整函数(舍弃小数部分，只保留整数部分)
            例如:np.floor(191.25) -> 191.0

        .astype(np.int32):数据类型强制转换。将默认的浮点数类型(Float)强制转换为32位有符号整数类型(Int32)
        '''

        return action_tokens

    def detokenize(self, action_tokens):
        # 将离散的Token解码回连续动作
        '''
        Input:
            action_tokens: np.ndarray 或 int
                           输入的离散动作代号(Token)，取值范围为整型 [0, vocab_size - 1]。
                           例如:191 或 np.array([191, 102, 242])
        Return:
            action_continuous: np.ndarray 或 float
                               还原后的物理世界连续动作数值，取值范围为浮点型 [action_min, action_max]。
                               例如:0.496 或 np.array([0.496, -0.203, 0.894])
        '''

        # 1.映射回[0, 1]
        action_norm = (action_tokens + 0.5) / self.vocab_size
        # 0.5的含义是把数据放到格子的中间，这样就保证了误差在格子大小的一半之内

        # 2.反归一化到[min, max]
        action_continuous = action_norm * (self.max - self.min) + self.min

        return action_continuous

tokenizer = RT1ActionTokenzizer()
raw_action = np.array([0.5, -0.2, 0.9]) # x, y, z速度
tokens = tokenizer.tokenize(raw_action)
detokens = tokenizer.detokenize(tokens)

print(f"原始动作: {raw_action}")

print(f"Token化结果: {tokens}") 
# 输出示例: [191, 102, 242] -> 这些整数直接作为 Transformer 的输入 Target

print(f"detoken化结果: {detokens}")
# 输出结果距离原始动作很近，但是不完全是原始动作
























