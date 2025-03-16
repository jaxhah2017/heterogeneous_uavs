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
        
    
    def estimate_chan_gain(self, d_ground: ndarray, h_uav: float) -> ndarray:
        """计算信道增益"""
        # 计算距离
        d_link = np.sqrt(np.square(d_ground) + np.square(h_uav))
        # 计算Los概率
        p_los = 1 / (1 + self.a * np.exp(-self.b * (180 / np.pi * np.arcsin(h_uav / d_link) - self.a)))
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

def calculate_zf_beamforming(H):
    n_uav, n_gt, Nr, Nt = H.shape
    
    # 初始化零迫波束赋形向量
    b_p = np.zeros((n_uav, n_gt, Nt), dtype=complex)

    for k in range(n_uav):
        for i in range(n_gt):
            h_ki = H[k, i].flatten()  # 获取用户 i 的信道向量
            print(f"\nh_ki for user {i}:")
            print(h_ki)
            
            # 构建非目标用户的信道矩阵 H_-i
            # if i > 0:
            #     H_neg_i = np.concatenate([H[k, :i], H[k, i+1:]], axis=0).reshape(-1, Nt)
            # else:
            #     H_neg_i = H[k, i+1:].reshape(-1, Nt)
            # print(f"H_neg_i for user {i}:")
            # print(H_neg_i)
            # 构建非目标用户的信道矩阵 H_neg_i（仅包含非零用户）
            non_zero_users = []
            for j in range(n_gt):
                if j != i and not np.allclose(H[k, j], 0):  # 排除当前用户和零用户
                    non_zero_users.append(H[k, j])
            if non_zero_users:
                H_neg_i = np.concatenate(non_zero_users, axis=0).reshape(-1, Nt)
            else:
                H_neg_i = np.empty((0, Nt))  # 空矩阵


            # 计算投影矩阵 P
            # if H_neg_i.size > 0:
            #     H_neg_i_H = H_neg_i.conj().T @ H_neg_i
            #     print("Marix shape:", H_neg_i_H.shape)
            #     # if np.linalg.matrix_rank(H_neg_i_H) < H_neg_i_H.shape[0]: # det(A) = 0
            #     #     print("Warning: Matrix is singular or nearly singular.")
            #     #     continue
                
            #     P = np.eye(Nt) - H_neg_i @ np.linalg.pinv(H_neg_i_H) @ H_neg_i.conj().T
            #     print(f"Projection matrix P for user {i}:")
            #     print(P)
            # else:
            #     P = np.eye(Nt)
            
            # 计算投影矩阵 P
            if H_neg_i.size > 0:
                # 转置 H_neg_i 为 (Nt, M)
                H_neg_i = H_neg_i.T  
                H_neg_i_H = H_neg_i.conj().T @ H_neg_i  # 形状 (M, M)
                P = np.eye(Nt) - H_neg_i @ np.linalg.pinv(H_neg_i_H) @ H_neg_i.conj().T # 计算伪逆矩阵，会使得当 非目标用户数量 < 天线数量 时，零迫算法无法完全消除所有干扰
                # # 计算投影矩阵 P 的理论依据：
                # P = I - H_neg_i * pinv(H_neg_i_H) * H_neg_i^H
                # 该矩阵将发射向量投影到非目标用户的零空间，从而消除干扰。
                # 当 H_neg_i 的秩不足时，伪逆确保 P 的存在性。
            else:
                P = np.eye(Nt)

            # 计算零空间向量 h_perp
            h_perp = P @ h_ki
            print(f"h_perp for user {i}:")
            print(h_perp)
            
            # 归一化得到零迫波束赋形向量
            if np.linalg.norm(h_perp) != 0:
                b_p[k, i] = h_perp / np.linalg.norm(h_perp)
            else:
                b_p[k, i] = np.zeros(Nt, dtype=complex)
    
    return b_p

if __name__ == '__main__':
    np.random.seed(2)
    
    d = np.arange(0, 1000, 100)
    Nt = 2
    Nr = 1
    chan_model = AirToGroundChannel(Nt=Nt, Nr=Nr, apply_small_fading=True)

    # 多用户场景
    n_gt = 5
    H = np.zeros((1, n_gt, Nr, Nt), dtype=complex)
    for i in range(3):
        H[0, i] = chan_model.estimate_chan_gain(100 + i * 10, 100)[0]
    print("\n多用户信道矩阵 H:")
    print(H)
    print("H 的形状:", H.shape)

    # 假设 n_uav = 1，即只有一个无人机
    n_uav = 1
    H = H.reshape(n_uav, n_gt, Nr, Nt)

    # 计算零迫波束赋形向量
    b_p = calculate_zf_beamforming(H)
    print("\n零迫波束赋形向量 b_p:")
    print(b_p)
    print("b_p 的形状:", b_p.shape)

    """产生信道和公有信息基础向量"""
    # d = np.arange(0, 1000, 100)
    # Nt = 5
    # Nr = 1
    # chan_model = AirToGroundChannel(Nt=Nt, Nr=Nr, apply_small_fading=True)

    # # 单用户场景
    # g = chan_model.estimate_chan_gain(100, 100)[0]
    # print("单用户信道向量 g:")
    # print(g)
    # print("g 的形状:", g.shape)
    # print("xxxx:", np.linalg.norm(g))

    # # 匹配滤波波束向量 u
    # u = g / np.linalg.norm(g)
    # print("\n匹配滤波波束向量 u:")
    # print(u)
    # print("u 的形状:", u.shape)

    # # 多用户场景
    # n_gt = 3
    # H = np.array([chan_model.estimate_chan_gain(100 + i * 10, 100)[0] for i in range(n_gt)])  # 修改信道矩阵生成方式
    # print("\n多用户信道矩阵 H:")
    # print(H)
    # print("H 的形状:", H.shape)

    # P_total = 10
    # # 公共流波束向量 uc
    # sum_h = np.sum(H, axis=0)
    # if np.allclose(sum_h, 0):
    #     print("Warning: sum_h is close to zero vector.")
    # else:
    #     uc = sum_h / np.linalg.norm(sum_h)
    #     print("\n公共流波束向量 uc:")
    #     print(uc)
    #     print("uc 的形状:", uc.shape)

    #     p_common = np.sqrt(P_total) * uc
    #     print("\n多用户公共流发射信号 p_common:")
    #     print(p_common)

    #     # 验证各用户的接收功率 |h_i^H * uc|^2
    #     received_powers = []
    #     for h_i in H:
    #         print("h_i_shape:", h_i.shape)
    #         print("原来信道:", h_i)
    #         x = h_i.conj().T
    #         print("信道共轭转置:", x)
    #         print("h_i_conj.T.shape:", x.shape)
    #         print("p_common_shape:", p_common.shape)
    #         power = np.abs(h_i.conj().T @ p_common) ** 2
    #         received_powers.append(power)


    #     print("\n各用户接收功率:")
    #     for i, power in enumerate(received_powers):
    #         print(f"用户 {i+1} 的接收功率: {power}")


