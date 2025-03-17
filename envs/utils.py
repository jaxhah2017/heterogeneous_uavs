import torch
import numpy as np
import random


def set_randseed(seed=10):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_jain_fairness_index(x):
    """Computes the Jain's fairness index of entries in given ndarray."""
    if x.size > 0:
        x = np.clip(x, 1e-6, np.inf)
        return np.square(x.sum()) / (x.size * np.square(x).sum())
    else:
        return 1

x = np.array([1, 1, 1, 1])

print(compute_jain_fairness_index(x))

class A_MAB_weight_opt:
    # TODO: 多臂老虎机动态更新权重更新
    def __init__(self, epsilon):
        self.epsilon = epsilon

        # 预先生成W个权重
        self.w = np.array([1/3, 1/3, 1/3], 
                          [0.33, 0.34, 0.33],
                          [0.32, 0.35, 0.33],
                          [0.35, 0.31, 0.34],
                          [0.32, 0.32, 0.36])

        self.W = len(self.w)

        # 预先生成初始奖励
        self.Reward_w = [0.5 for _ in range(self.W)]
    def get_weight(self):
        if random.random() > self.epsilon:
            w_choose = np.random.randint(0, self.W)
        else:
            w_choose = np.argmax(self.Reward_w)

        return self.w[w_choose]
    