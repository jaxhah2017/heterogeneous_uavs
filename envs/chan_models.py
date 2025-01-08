import numpy as np
from numpy import ndarray


class AirToGroundChannel(object):
    """Air-to-ground (ATG) channel model proposed in "Optimal LAP Altitude for Maximum Coverage".
    Parameter a and b are provided in paper
    "Efficient 3-D Placement of an Aerial Base Station in Next Generation Cellular Networks".
    https://github.com/zhangxiaochen95/cross_layer_opt_with_grl/blob/main/envs/chan_models.py provides this code.
    """

    ATG_CHAN_PARAMS = {
        'suburban': {'a': 4.88, 'b': 0.43, 'eta_los': 0.1, 'eta_nlos': 21},
        'urban': {'a': 9.61, 'b': 0.16, 'eta_los': 1, 'eta_nlos': 20},
        'dense-urban': {'a': 12.08, 'b': 0.11, 'eta_los': 1.6, 'eta_nlos': 23},
        'high-rise-urban': {'a': 27.23, 'b': 0.08, 'eta_los': 2.3, 'eta_nlos': 34}
    }
    
    def __init__(self, scene: str = 'urban', fc: float = 2.4e9, Nt: int = 2, Nr: int = 1, apply_small_fading: bool = False) -> None:
        for k, v in self.ATG_CHAN_PARAMS[scene].items():
            self.__setattr__(k, v)
        self.fc = fc
        self.Nt = Nt
        self.Nr = Nr
        self.apply_small_fading = apply_small_fading
        
    
    def estimate_chan_gain(self, d_ground: ndarray, h_ubs: float) -> ndarray:
        """计算信道增益"""
        # 计算距离
        d_link = np.sqrt(np.square(d_ground) + np.square(h_ubs))
        # 计算Los概率
        p_los = 1 / (1 + self.a * np.exp(-self.b * (180 / np.pi * np.arcsin(h_ubs / d_link) - self.a)))
        # 计算自由空间路径损失
        fspl = (4 * np.pi * self.fc * d_link / 3e8) ** 2
        # 计算路径损失
        pl = p_los * fspl * 10 ** (self.eta_los / 10) + (1 - p_los) * fspl * 10 ** (self.eta_nlos / 10)
        # 大尺度衰落
        h_large = np.sqrt(1 / pl)
        # 信道矩阵
        H = np.ones((self.Nr, self.Nt)) * h_large
        # 小尺度衰落
        h_small = (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt)) / np.sqrt(2)

        if self.apply_small_fading:
            H = H * h_small

        return H

if __name__ == '__main__':
    d = np.arange(0, 1000, 100)
    # print(d)
    Nt = 5
    Nr = 1
    chan_model = AirToGroundChannel(Nt=Nt, Nr=Nr)
    H1 = chan_model.estimate_chan_gain(100, 100)
    print(H1)
    print(H1.shape)


    Pc = np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)  # 公共流预编码向量
    P1 = np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)  # 用户 1 的私有流预编码向量
    P2 = np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)  # 用户 2 的私有流预编码向量
    
    # H1T = H1.conj().T
    # print(H1T.shape)
    # print(Pc.shape)
    # pt = np.abs(H1T @ Pc) ** 2

    # print(H1)

    # print() 

    # print(pt)
    