import numpy as np

# 假设参数
Nt = 4  # 基站天线数
N_users = 3  # 用户数
alpha = 2  # 路径损耗指数

# 用户与基站的距离
distances = np.array([100, 150, 200])  # 用户 1、2、3 的距离（单位：米）

# 计算路径损耗
path_losses = 1 / (distances ** alpha)  # 路径损耗

# 生成信道矩阵（仅考虑大尺度衰落）
H = np.sqrt(path_losses).reshape(-1, 1) * np.ones((N_users, Nt))  # 信道矩阵 (N_users x Nt)

print('H:', H)
print('H shape:', H.shape)

# 随机生成预编码矩阵
Pc = np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)  # 公共流预编码向量 (Nt x 1)
P1 = np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)  # 用户 1 的私有流预编码向量 (Nt x 1)
P2 = np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)  # 用户 2 的私有流预编码向量 (Nt x 1)
P3 = np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)  # 用户 3 的私有流预编码向量 (Nt x 1)

# 组合预编码矩阵
P = np.hstack([Pc, P1, P2, P3])  # 预编码矩阵 (Nt x 4)

# 噪声功率
sigma_sq = 0.1  # 噪声功率

# 计算 SINR
def calculate_sinr(H, P, sigma_sq):
    N_users, Nt = H.shape
    SINR_c = np.zeros(N_users)  # 存储每个用户的公共流 SINR
    SINR_p = np.zeros(N_users)  # 存储每个用户的私有流 SINR

    for k in range(N_users):
        h_k = H[k, :].reshape(-1, 1)  # 用户 k 的信道向量 (Nt x 1)
        p_c = P[:, 0].reshape(-1, 1)  # 公共流预编码向量 (Nt x 1)
        p_k = P[:, k + 1].reshape(-1, 1)  # 用户 k 的私有流预编码向量 (Nt x 1)

        # 计算公共流 SINR
        signal_power_c = np.abs(h_k.conj().T @ p_c) ** 2
        interference_power_c = 0
        for j in range(1, 4):  # 私有流的干扰
            p_j = P[:, j].reshape(-1, 1)
            interference_power_c += np.abs(h_k.conj().T @ p_j) ** 2
        SINR_c[k] = signal_power_c / (interference_power_c + sigma_sq)

        # 计算私有流 SINR
        signal_power_p = np.abs(h_k.conj().T @ p_k) ** 2
        interference_power_p = 0
        for j in range(1, 4):  # 其他用户的私有流干扰
            if j != k + 1:
                p_j = P[:, j].reshape(-1, 1)
                interference_power_p += np.abs(h_k.conj().T @ p_j) ** 2
        SINR_p[k] = signal_power_p / (interference_power_p + sigma_sq)

    return SINR_c, SINR_p

# 计算 SINR
SINR_c, SINR_p = calculate_sinr(H, P, sigma_sq)
print("每个用户的公共流 SINR:")
print(SINR_c)
print("每个用户的私有流 SINR:")
print(SINR_p)