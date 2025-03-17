import numpy as np

from maps import Maps

from chan_models import *

import random

from utils import *

class MultiUAVEnvironment(object):
    
    h_uav = 100 # 飞行高度100m
    n0 = 1e-3 * np.power(10, -170 / 10) # 噪声
    fc = 2.4e9  # 载频
    scene = 'dense-urban'  # 信道场景
    safe_dist = 10  # 安全距离（m）
    bw = 180e3 # 频带
    d_min = 5 # 无人机最小安全距离 5m

    def __init__(self, args) -> None:
        super(MultiUAVEnvironment, self).__init__()
        self.args = args
        self.collision_penlty = args.collision_penlty
        # 初始map信息
        self.map = Maps[args.map]
        self.map.set_positions()
        map_params = self.map.get_params()
        print(map_params)
        for k, v in map_params.items():  
            setattr(self, k, v)  # 初始化: range_pos, n_uav, n_gt, n_eve, pos_uav, pos_gt, pos_eve

        # 天线数量
        self.Nt = args.Nt
        self.Nr = 1

        # 距离矩阵
        self.dis_U2G = np.zeros((self.n_uav, self.n_gt))
        self.dis_U2E = np.zeros((self.n_uav, self.n_eve))
        self.dis_U2U = np.zeros((self.n_uav, self.n_uav))

        # 关联矩阵
        self.cov_U2G = np.zeros((self.n_uav, self.n_gt))
        self.cov_U2U = np.zeros((self.n_uav, self.n_uav))
        self.sche_U2G = np.zeros((self.n_uav, self.n_gt))
        self.cov_U2E = np.zeros((self.n_uav, self.n_eve))

        self.t = 0 # 时间步

        self.H_U2G = np.zeros((self.n_uav, self.n_gt, self.Nr, self.Nt), dtype=complex) # 信道矩阵
        self.H_U2G_norm = np.zeros((self.n_uav, self.n_gt)) # 信道范数
        self.H_U2E = np.zeros((self.n_uav, self.n_eve, self.Nr, self.Nt), dtype=complex) # 信道矩阵
        self.H_U2E_norm = np.zeros((self.n_uav, self.n_eve)) # 信道范数

        self.chan_model = AirToGroundChannel(scene=args.scene, 
                                             fc=args.fc, 
                                             Nt=args.Nt, 
                                             Nr=self.Nr, 
                                             apply_small_fading=args.apply_small_fading) # 信道   

        self.a_mab = A_MAB_weight_opt(args.mab_epsilon)
        
        self.shannon_capacity_c = np.zeros((self.n_gt))
        self.shannon_capacity_p = np.zeros((self.n_gt))
        self.shannon_capacity_c_k = np.zeros(self.n_uav)
        self.shannon_capacity_c_e_k = np.zeros((self.n_eve, self.n_uav))
        self.shannon_capacity_p_e_i = np.zeros((self.n_eve, self.n_gt))

        self.sr_c = 0
        self.sr_p = 0
        self.power_consumption_k_t = np.zeros(self.n_uav) # 到t为止的累积能耗

        self.avoid_collision = args.avoid_collision


        self.V_max = args.v_max
        self.P_max = args.P_max

        self.reward = np.zeros((self.n_uav))
        self.reward_scale = 0.1
        self.fair_index = 0.0
        self.avg_epi_rate_gt = np.zeros((self.n_gt), dtype=np.float32)
        self.rate_gt_t = np.zeros((self.n_gt), dtype=np.float32)
        self.rate_ubs_t = np.zeros((self.n_uav), dtype=np.float32)

    def reset(self) -> None:
        self.t = 0
        self.fair_index = 0.0
        self.avg_epi_rate_gt = np.zeros((self.n_gt), dtype=np.float32)
        self.power_consumption_k_t = np.zeros(self.n_uav) # 能耗清零
        self.reward = np.zeros((self.n_uav)) # 奖励清零
    
    def step(self, actions) -> None:
        self.t = self.t + 1

        # 获取行为
        velocity = actions['velocity'] # 速度  shape: n_uav * 1
        direction = actions['direction'] # 方向 shape: n_uav * 1
        alpha = actions['alpha'] # 控制公共信息的功率[0, Pc_max] shape: n_uav * 1 不一定所有无人机都有人服务
        theta = actions['theta']  # 微调公有信息的波束方向[0, 2pi] shape: n_uav * 1 不一定所有无人机都有人服务 
        beta = actions['beta'] # 控制私有信息的功率[0, Pp_max] shape: n_uav * n_gt 不一定所有无人机都有人服务
        phi = actions['phi'] # 微调私有信息的波束方向[0, 2pi] shape:  n_uav * n_gt 不一定所有无人机都有人服务

        self.uav_move_and_energy_model(velocity, direction) # 运动更新位置，计算推进能耗，
        
        self.update_dis_conn() # 更新距离、生成信道、碰撞检测

        self.transmit_data(alpha, theta, beta, phi) # 传输数据

        self.secrecy_rate_model() # 安全模型，计算安全相关内容

        w = self.a_mab.get_weight()

        reward = self.get_reward(w, self.reward_scale)
        

    def update_dis_conn(self) -> None:
        """更新距离与关联模型"""
        gt_becov = [[] for i in range(self.n_gt)]  # UAV k 覆盖的GT
        self.dis_U2G = np.zeros((self.n_uav, self.n_gt))
        self.cov_U2G = np.zeros((self.n_uav, self.n_gt))
        for k in range(self.n_uav):
            for i in range(self.n_gt):
                self.dis_U2G[k][i] = np.linalg.norm(self.pos_uav[k] - self.pos_gt[i])
                self.cov_U2G[k][i] = 1 if self.dis_U2G[k][i] <= self.cov_range[k] else 0  # 覆盖关系 
                gt_becov[i].append(k) if self.cov_U2G[k][i] == 1 else None

        # UAV与Eve
        self.cov_U2E = np.zeros((self.n_uav, self.n_eve))
        self.dis_U2E = np.zeros((self.n_uav, self.n_eve))
        for k in range(self.n_uav):
            for e in range(self.n_eve):
                self.dis_U2E[k][e] = np.linalg.norm(self.pos_uav[k] - self.pos_eve[e])
                self.cov_U2E[k][e] = 1 if self.dis_U2E[k][e] <= self.cov_range[k] else 0 # 窃听关系
                
        # UAV与UAV
        self.dis_U2U = np.zeros((self.n_uav, self.n_uav))
        self.cov_U2U = np.zeros((self.n_uav, self.n_uav))
        for k in range(self.n_uav):
            for l in range(self.n_uav):
                self.dis_U2U[k][l] = np.linalg.norm(self.pos_uav[k] - self.pos_uav[l])
                self.cov_U2U[k][l] = 1 if self.dis_U2U[k][l] <= self.cov_range[k] else 0 # 通信关联

        """生成信道"""
        self.generate_channel()

        # GT与覆盖自己并且信道质量最好的无人机进行通信
        self.sche_U2G = np.zeros((self.n_uav, self.n_gt))
        self.uav_serv_gt = [[] for _ in range(self.n_uav)]
        for i in range(self.n_gt):
            gt_becov[i] = sorted(gt_becov[i], key=lambda k: self.H_U2G_norm[k][i], reverse=True)
            for k in gt_becov[i]:
                if sum(self.sche_U2G[k]) < self.serv_capacity[k]:
                    self.sche_U2G[k][i] = 1 # UAV k服务GT i
                    self.uav_serv_gt[k].append(i)
                    break

        """碰撞检测"""
        self.mask_collision = ((self.dis_U2U + 99999 * np.eye(self.n_uav)) < self.safe_dist).any(1)

    def generate_channel(self) -> None:
        # 生成UAV与GT的信道
        self.H_U2G = np.zeros((self.n_uav, self.n_gt, self.Nr, self.Nt), dtype=complex) # 信道矩阵
        self.H_U2G_norm = np.zeros((self.n_uav, self.n_gt)) # 信道范数
        for k in range(self.n_uav):
            for i in range(self.n_gt):
                if self.cov_U2G[k][i] == 1: # 只要在覆盖范围中就都生成信道，因为在覆盖范围中就会有影响
                    g = self.chan_model.estimate_chan_gain(d_ground=self.dis_U2G[k][i], h_uav=self.h_uav)[0]
                    self.H_U2G[k][i] = g
                    self.H_U2G_norm[k][i] = np.linalg.norm(g) # 计算信道范数，通过信道范数比较信道质量

        # 生成UAV与Eve的信道
        self.H_U2E = np.zeros((self.n_uav, self.n_eve, self.Nr, self.Nt), dtype=complex) # 信道矩阵
        self.H_U2E_norm = np.zeros((self.n_uav, self.n_eve)) # 信道范数
        for k in range(self.n_uav) :
            for e in range(self.n_eve):
                if self.cov_U2E[k][e] == 1:
                    g = self.chan_model.estimate_chan_gain(d_ground=self.dis_U2E[k][e], h_uav=self.h_uav)[0]
                    self.H_U2E[k][e] = g 

    def calculate_zf_beamforming(self, H):
        n_uav, n_gt, Nr, Nt = H.shape
        
        # 初始化零迫波束赋形向量
        b_p = np.zeros((n_uav, n_gt, Nt), dtype=complex)

        for k in range(n_uav):
            for i in self.uav_serv_gt[k]:
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
                for j in self.uav_serv_gt[k]:
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

    def uav_move_and_energy_model(self, velocity, direction):
        """运动"""
        # 计算位移
        dx = velocity * np.cos(direction)
        dy = velocity * np.sin(direction)
        # 更新无人机位置
        self.pos_uav = self.pos_uav + np.column_stack((dx, dy))
        self.pos_ubs = np.clip(self.pos_ubs,
                               a_min=0,
                               a_max=self.range_pos)

        """能耗模型"""
        # 定义推进能耗相关的常数参数
        P0 = 50  # 基本功率消耗
        P_ind = 30  # 诱导功率消耗
        U_tip = 75  # 翼尖速度
        v0 = 7.5  # 特征速度
        d0 = 0.05  # 阻力系数
        rho = 1.225  # 空气密度，kg/m^3
        s_sol = 0.8  # 固体分数
        A_area = 1.0  # 迎风面积
        # 计算每架无人机的推进能耗
        for k in range(self.n_uav):
            vk = velocity[k]  # 第k架无人机的速度

            # 根据公式计算功率消耗
            term1 = P0 * (1 + 3 * vk**2 / U_tip**2)
            term2 = P_ind * ((1 + vk**4 / (4 * v0**4))**0.5 - vk**2 / (2 * v0**2))**0.5
            term3 = 0.5 * d0 * rho * s_sol * A_area * vk**3
            self.power_consumption_k_t[k] += term1 + term2 + term3
    
    def transmit_data(self, alpha, theta, beta, phi) -> None:
        """生成基础波束方向"""
        # 生成公有信息的波束方向
        P_comm = np.zeros((self.n_uav, self.Nr, self.Nt), dtype=complex)
        for k in range(self.n_uav):
            if len(self.uav_serv_gt[k]) == 0: # 如果没有服务的用户，那么为0即可
                continue
            H = self.H_U2G[k][self.uav_serv_gt[k]] # 无人机h的信道矩阵
            sum_H = np.sum(H, axis=0)
            base_beam = sum_H / np.linalg.norm(sum_H) # 启发式基础束
            P_comm[k] = np.sqrt(alpha[k]) * base_beam * np.exp(1j * theta[k]) # 产生波束
        # 生成私有信息的波束方向
        P_priv = np.zeros((self.n_uav, self.n_gt, self.Nr, self.Nt), dtype=complex)
        zf_beam = self.calculate_zf_beamforming(self.H_U2G)
        for k in range(self.n_uav):
            for i in self.uav_serv_gt[k]:
                P_priv[k][i] = np.sqrt(beta[k][i]) * zf_beam[k][i] * np.exp(1j * phi[k][i])

        def shannon_capacity(self, s, n):
            # 计算香农容量 (Mbps)
            return self.bw * np.log(1 + s / n) * 1e-6

        """传输数据"""
        # 传输公有信息
        self.shannon_capacity_c = np.zeros((self.n_gt))
        I_i_o = np.zeros((self.n_gt))
        for k in range(self.n_uav):
            if len(self.uav_serv_gt[k]) == 0:
                continue
            Ip = 0
            # 计算私有干扰
            for i in self.uav_serv_gt[k]: # 计算当前无人机下所有人的私有干扰
                Ip += np.linalg.norm(self.H_U2G[k][i].conj().T @ P_priv[k][i]) ** 2
            for i in self.uav_serv_gt[k]: # 遍历当前无人机下服务的所有用户
                # 计算其他无人机对其的干扰
                for l in range(self.n_uav):
                    if l != k and self.cov_U2G[l][i] == 1: # 其他覆盖GT i的无人机会造成干扰
                        I_i_o[i] += np.linalg.norm(self.H_U2G[l][i].conj().T @ P_comm[l]) ** 2 # 公有干扰
                        for j in self.uav_serv_gt[l]: # 私有干扰
                            I_i_o[i] += np.linalg.norm(self.H_U2G[l][i].conj().T @ P_priv[l][j]) ** 2
                s = np.linalg.norm(self.H_U2G[k][i].conj().T @ P_comm[k]) ** 2
                self.shannon_capacity_c[i] = shannon_capacity(s, I_i_o[i] + Ip + self.n0)
        
        self.shannon_capacity_p = np.zeros((self.n_gt)) 
        # 传输私有信息
        for k in range(self.n_uav):
            if len(self.uav_serv_gt[k]) == 0:
                continue
            for i in self.uav_serv_gt[k]:
                I_op = 0
                for j in self.uav_serv_gt[k]:
                    if j != i:
                        I_op += np.linalg.norm(self.H_U2G[k][j].conj().T @ P_priv[k][j]) ** 2
                s = np.linalg.norm(self.H_U2G[k][i].conj().T @ P_priv[k][i]) ** 2
                self.shannon_capacity_p[i] = shannon_capacity(s, I_i_o[i] + I_op + self.n0)

        # Eve窃听
        # 窃听公有信息
        self.shannon_capacity_c_e_k = np.zeros((self.n_eve, self.n_uav))
        I_e_k_o = np.zeros((self.n_eve, self.n_uav))
        for e in range(self.n_eve):
            for k in range(self.n_uav):
                if self.cov_U2E[k][e] == 0:
                    continue
                s = np.linalg.norm(self.H_U2E[k][e].conj().T @ P_comm[k]) ** 2
                # 计算当前无人机下的私有
                # 计算私有干扰
                Ip = 0
                for i in self.uav_serv_gt[k]:
                    Ip += np.linalg.norm(self.H_U2E[k][e].conj().T @ P_priv[k][i]) ** 2

                for l in range(self.n_uav):
                    if l != k and self.cov_U2E[l][e] == 1:
                        I_e_k_o[e][k] += np.linalg.norm(self.H_U2E[l][e].conj().T @ P_comm[l]) ** 2
                        for j in self.uav_serv_gt[l]:
                            I_e_k_o[e][k] += np.linalg.norm(self.H_U2E[l][e].conj().T @ P_priv[l][j]) ** 2
                self.shannon_capacity_c_e_k[e][k] = shannon_capacity(s, Ip + I_e_k_o[e][k] + self.n0)
        # 窃听私有信息
        self.shannon_capacity_p_e_i = np.zeros((self.n_eve, self.n_gt))
        for e in range(self.n_eve):
            for k in range(self.n_uav):
                if self.cov_U2E[k][e] == 0:
                    continue
                for i in self.uav_serv_gt[k]:
                    s = np.linalg.norm(self.H_U2E[k][e].conj().T @ P_priv[k][i]) ** 2
                    Ip = 0
                    for j in self.uav_serv_gt[k]: # 当前无人机下其他人的是私有信息干扰
                        if j != i:
                            Ip += np.linalg.norm(self.H_U2E[k][e].conj().T @ P_priv[k][j]) ** 2 
                    Ip += np.linalg.norm(self.H_U2E[k][e].conj().T @ P_comm[k]) ** 2 # 当前无人机的公有信息干扰
                    self.shannon_capacity_p_e_i[e][i] = shannon_capacity(s, Ip + I_e_k_o[e][k] + self.n0)

        self.rate_gt_t = self.shannon_capacity_c + self.shannon_capacity_p

        self.avg_epi_rate_gt = (self.avg_epi_rate_gt * self.t + self.rate_gt_t) / (self.t + 1)

        self.fair_index = compute_jain_fairness_index(self.avg_epi_rate_gt)

    def secrecy_rate_model(self):
        """安全模型"""
        self.sr_c = 0
        self.shannon_capacity_c_k = np.zeros(self.n_uav)
        for k in range(self.n_uav):
            self.shannon_capacity_c_k[k] = np.min(self.shannon_capacity_c[self.uav_serv_gt[k]])
            e = np.argmax(self.shannon_capacity_c_e_k[:, k])
            r_c_e = self.shannon_capacity_c_e_k[e, k]
            self.sr_c += self.cov_U2E[k][e] * max(0, self.shannon_capacity_c_k[k] - r_c_e)

        self.sr_p = 0
        for k in range(self.uav):
            for i in self.uav_serv_gt[k]:
                r_p_i = self.shannon_capacity_p[i]
                e = np.argmax(self.shannon_capacity_p_e_i[:, i])
                r_p_e_i = self.shannon_capacity_p_e_i[e, i]
                self.sr_p += self.cov_U2E[k][e] * max(0, r_p_i - r_p_e_i)

    def get_obs(self) -> list:
        return [self.get_obs_agent(agent_id=agent_id) for agent_id in range(self.uav)]

    def get_obs_agent(self, agent_id: int) -> dict:
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        uav_feats = np.zeros(self.obs_uav_feats_size, dtype=np.float32)
        gt_feats = np.zeros(self.obs_gt_feats_size, dtype=np.float32)
        eve_feats = np.zeros(self.obs_eve_feats_size, dtype=np.float32)

        own_feats[0:2] = self.pos_uav[agent_id] / self.range_pos
        own_feats[2] = self.shannon_capacity_c_k[agent_id] / self.shannon_capacity_c_max # self.shannon_capacity_c_max TODO
        own_feats[3] = self.power_consumption_k_t[agent_id] / self.power_consumption_k_t_max # self.power_consumption_k_t_max TODO
        
        # 其他无人机的特征
        other_uav = [ubs_id for ubs_id in range(self.n_agents) if ubs_id != agent_id]
        for j, ubs_id in enumerate(other_uav):
            if self.cov_U2U[agent_id][ubs_id]:
                uav_feats[j, 0:2] = (self.pos_uav[ubs_id] - self.pos_uav[agent_id]) / self.range_pos  # relative pos

        # 覆盖范围内GT的特征
        for i in range(self.n_gt):
            if self.cov_U2G[agent_id][i]:
                gt_feats[i, 0:2] = (self.pos_gt[i] - self.pos_ubs[agent_id]) / self.range_pos  # relative pos
                gt_feats[i, 2] = self.shannon_capacity_p[i] / self.shannon_capacity_p_max # self.shannon_capacity_p_max TODO
        
        # 覆盖范围内Eve的特征
        for e in range(self.n_eve):
            if self.cov_U2E[agent_id][e]:
                eve_feats[e, 0:2] = (self.pos_eve[e] - self.pos_ubs[agent_id]) / self.range_pos  # relative pos
    def get_obs_size(self) -> dict:

        return dict(agent=self.obs_own_feats_size, 
                    ubs=self.obs_uav_feats_size, 
                    gt=self.obs_gt_feats_size,
                    eve=self.obs_eve_feats_size)

    def get_env_info(self):
        obs_size = self.get_obs_size()
        gt_feats_size = obs_size['gt']
        other_feats_size = obs_size['agent'] + np.prod(obs_size['ubs']) + np.prod(obs_size['eve'])

        env_info = dict(n_uav=self.n_uav,
                        gt_feats_size=gt_feats_size,
                        other_feats_size=other_feats_size,
                        episode_limit=self.episode_limit)

    @property
    def obs_own_feats_size(self) -> int:
        """
        - UAV本身的坐标 2 (x, y) 
        - UAV本身的能耗 1 
        - 当前无人机服务下的公有信息容量 R_k^c 1
        """
        o_fs = 2 + 1 + 1

        return o_fs

    @property
    def obs_uav_feats_size(self) -> int:
        """
        - 附近UAV的相对坐标 (x, y)
        """
        u_fs = 2

        return self.n_uav - 1, u_fs
    
    @property
    def obs_gt_feats_size(self) -> int:
        """
        - 附近GT的坐标 (x, y)
        - 每个GT的私有信息容量 
        """
        g_fs = 2 + 1

        return self.n_gt, g_fs

    @property
    def obs_eve_feats_size(self) -> int:
        """
        - 附近Eve的坐标 (x, y)
        """

        e_fs = 1

        return self.n_eve, e_fs


    def get_state_size(self) -> int:
        """
        所有UAV的位置
        所有GT的位置
        所有EVE的位置
        无人机的行为状态
        ... TODO
        """

        pass

    def get_reward(self, reward_scale, w = [1/3, 1/3, 1/3]):

        power_consumption_t = np.sum(self.power_consumption_k_t)

        uav_rewards = (w[0] * (self.sr_c + self.sr_p) / self.secrecy_rate_max # TODO self.secrecy_rate_max
                       + w[1] * self.fair_index 
                       - w[2] * power_consumption_t / self.power_consumption_t_max) # TODO self.power_consumption_t_max

        idle_ubs_mask = np.array([len(self.uav_serv_gt[k]) == 0 for k in range(self.n_uav)])

        uav_rewards = uav_rewards * (1 - idle_ubs_mask)

        if self.avoid_collision:
            uav_rewards = uav_rewards - self.mask_collision * self.collision_penlty 

        uav_rewards = uav_rewards * reward_scale

        return uav_rewards
        
    
if __name__ == '__main__':
    set_randseed(seed=2)

    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_types', type=int, default=2, help='the number of type for UAVs')
    parser.add_argument('--Nt', type=int, default=2, help='the number of transmit antennas > 1')
    parser.add_argument('--apply_small_fading', type=bool, default=True, help='where use small fading')
    parser.add_argument('--scene', type=str, default='urban', help='environmental scenes')
    parser.add_argument('--fc', type=float, default=2.4e9, help='carrier frequency')
    parser.add_argument('--map', type=str, default='2uav', help='map type')
    parser.add_argument('--avoid_collision', type=bool, default=True, help='avoid UAV collission')
    parser.add_argument('--collision_penlty', type=int, default=5, help='UAV collission penlty')
    parser.add_argument('--V_max', type=float, default=25, help='m/s the limits of velocity on UAVs')
    parser.add_argument('--P_max', type=float, default=35, help='Watt the limits of Power on UAVs')
    parser.add_argument('--mab_epsilon', type=float, default=0.99, help='epsilon of MAB')
    parser.add_argument('--episode_limit', type=int, default=60, help='max episode length')
    
    
    args = parser.parse_args()
    
    env = MultiUAVEnvironment(args=args)

    # velocity = actions['velocity'] # 速度  shape: n_uav * 1
    # direction = actions['direction'] # 方向 shape: n_uav * 1
    # alpha = actions['alpha'] # 控制公共信息的功率[0, Pc_max] shape: n_uav * 1 不一定所有无人机都有人服务
    # theta = actions['theta']  # 微调公有信息的波束方向[0, 2pi] shape: n_uav * 1 不一定所有无人机都有人服务 
    # beta = actions['beta'] # 控制私有信息的功率[0, Pp_max] shape: n_uav * n_gt 不一定所有无人机都有人服务
    # phi = actions['pho'] # 微调私有信息的波束方向[0, 2pi] shape:  n_uav * n_gt 不一定所有无人机都有人服务
    actions = dict()
    n_uav = 2
    n_gt = 7
    velocity = np.array(np.random.uniform(0, 20, n_uav))
    direction = np.array(np.random.uniform(0, 2 * np.pi, n_uav))
    alpha = np.array(np.random.uniform(0, 15, n_uav))
    theta = np.array(np.random.uniform(0, 2 * np.pi, n_uav))
    beta = np.array(np.random.uniform(0, 10, (n_uav, n_gt)))
    beta[0, 2:] = 0
    phi = np.array(np.random.uniform(0, 2 * np.pi, (n_uav, n_gt)))
    beta[0, 2:] = 0

    actions = dict(velocity=velocity, direction=direction, alpha=alpha, theta=theta, beta=beta, phi=phi)

    env.step(actions=actions)
    # print(env.__dict__)
    
    # env.update_dis_conn()

    
    
    

    

    