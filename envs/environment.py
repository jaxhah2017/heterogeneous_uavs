import numpy as np

from maps import MAPS

from chan_models import AirToGroundChannel

from het_uavs import HetergeneousUav

import random

from utils import *

class MultiUAVEnvironment(object):
    
    h_ubs = 100 # 飞行高度100m
    n0 = 1e-3 * np.power(10, -170 / 10) # 噪声
    fc = 2.4e9  # 载频
    scene = 'dense-urban'  # 信道场景
    safe_dist = 10  # 安全距离（m）

    def __init__(self, args) -> None:
        super(MultiUAVEnvironment, self).__init__()
        self.args = args
        self.chan_model = AirToGroundChannel(scene=args.scene, 
                                             fc=args.fc, 
                                             Nt=args.Nt, 
                                             Nr=args.Nr, 
                                             apply_small_fading=args.apply_small_fading) # 信道
        self.het_uavs = HetergeneousUav(n_uav=args.n_uav, n_types=args.n_types) # 初始化异构UAV的特性
        self.uavs_type = [random.randint(0, 1) for _ in range(args.n_uav)]  # 随机初始化UAV类型
        self.serv_capacity = [self.het_uavs.service_capacity[int(k)] for k in self.uavs_type] # 服务能力
        self.cov_capacity = [self.het_uavs.coverage_capacity[int(k)] for k in self.uavs_type] # 覆盖能力
        self.comm_capacity = [self.het_uavs.communication_range[int(k)] for k in self.uavs_type] # 通信范围

        # 数量
        self.n_uav = args.n_uav
        self.n_gt = args.n_gt
        self.n_eve = args.n_eve
        
        # 天线数量
        self.Nt = args.Nt
        self.Nr = args.Nr

        # 初始map信息
        self.pos_gts = []
        self.pos_ubs = []
        self.pos_eves = []
        self.range_pos = 0
        
        self.map = MAPS['{}uav'.format(self.n_uav)]
        self.map.set_positions()
        map_params = self.map.get_params()
        # print(map_params)
        for k, v in map_params.items():  
            setattr(self, k, v)  # 初始化: range_pos, n_ubs, n_gts, n_eves

        # 距离矩阵
        self.dis_U2G = np.zeros((self.n_uav, self.n_gt))
        self.dis_U2E = np.zeros((self.n_uav, self.n_eve))
        self.dis_U2U = np.zeros((self.n_uav, self.n_uav))

        # 关联矩阵
        self.cov_U2G = np.zeros((self.n_uav, self.n_gt))
        self.cov_U2U = np.zeros((self.n_uav, self.n_uav))
        self.sche_U2G = np.zeros((self.n_uav, self.n_gt))
        self.sche_U2E = np.zeros((self.n_uav, self.n_eve))

        self.t = 0 # 时间步

        self.H_U2G = np.zeros((self.n_uav, self.n_gt, self.Nr, self.Nt)) # 信道矩阵
        self.H_U2G_norm = np.zeros((self.n_uav, self.n_gt)) # 信道范数
        self.H_U2E = np.zeros((self.n_uav, self.n_eve, self.Nr, self.Nt)) # 信道矩阵
        self.H_U2E_norm = np.zeros((self.n_uav, self.n_eve)) # 信道范数

    def reset(self) -> None:
        self.t = 0
        # TODO
    
    def step(self, acts) -> None:
        pass

    def update_dis_conn(self) -> None:
        # UAV与GT
        gt_becov = [[] for i in range(self.n_gt)]  # UAV k 覆盖的GT
        self.dis_U2G = np.zeros((self.n_uav, self.n_gt))
        self.cov_U2G = np.zeros((self.n_uav, self.n_gt))
        for k in range(self.n_uav):
            for i in range(self.n_gt):
                self.dis_U2G[k][i] = np.linalg.norm(self.pos_ubs[k] - self.pos_gts[i])
                self.cov_U2G[k][i] = 1 if self.dis_U2G[k][i] <= self.cov_capacity[k] else 0  # 覆盖关系 
                gt_becov[i].append(k) if self.cov_U2G[k][i] == 1 else None

        # UAV与Eve
        self.sche_U2E = np.zeros((self.n_uav, self.n_eve))
        self.dis_U2E = np.zeros((self.n_uav, self.n_eve))
        for k in range(self.n_uav):
            for e in range(self.n_eve):
                self.dis_U2E[k][e] = np.linalg.norm(self.pos_ubs[k] - self.pos_eves[e])
                self.sche_U2E[k][e] = 1 if self.dis_U2E[k][e] <= self.cov_capacity[k] else 0 # 窃听关系
                
        # UAV与UAV
        self.dis_U2U = np.zeros((self.n_uav, self.n_uav))
        self.cov_U2U = np.zeros((self.n_uav, self.n_uav))
        for k in range(self.n_uav):
            for l in range(self.n_uav):
                self.dis_U2U[k][l] = np.linalg.norm(self.pos_ubs[k] - self.pos_ubs[l])
                self.cov_U2U[k][l] = 1 if self.dis_U2U[k][l] <= self.comm_capacity[k] else 0 # 通信关联

        # 生成信道
        self.generate_channel()

        # GT与覆盖自己并且信道质量最好的无人机进行通信
        self.sche_U2G = np.zeros((self.n_uav, self.n_gt))
        for i in range(self.n_gt):
            gt_becov[i] = sorted(gt_becov[i], key=lambda k: self.H_U2G_norm[k][i], reverse=True)
            for k in gt_becov[i]:
                if sum(self.sche_U2G[k]) < self.serv_capacity[k]:
                    self.sche_U2G[k][i] = 1 # UAV k服务GT i
                    break


        # print(self.__dict__)

    def generate_channel(self) -> None:
        # 生成UAV与GT的信道
        self.H_U2G = np.zeros((self.n_uav, self.n_gt, self.Nr, self.Nt)) # 信道矩阵
        self.H_U2G_norm = np.zeros((self.n_uav, self.n_gt)) # 信道范数
        for k in range(self.n_uav):
            for i in range(self.n_gt):
                if self.cov_U2G[k][i] == 1:
                    g = self.chan_model.estimate_chan_gain(d_ground=self.dis_U2G[k][i], h_ubs=self.h_ubs)[0]
                    self.H_U2G[k][i] = g
                    self.H_U2G_norm[k][i] = np.linalg.norm(g) # 计算信道范数，通过信道范数比较信道质量

        # 生成UAV与Eve的信道
        self.H_U2E = np.zeros((self.n_uav, self.n_eve, self.Nr, self.Nt)) # 信道矩阵
        self.H_U2E_norm = np.zeros((self.n_uav, self.n_eve)) # 信道范数
        for k in range(self.n_uav) :
            for e in range(self.n_eve):
                if self.sche_U2E[k][e] == 1:
                    g = self.chan_model.estimate_chan_gain(d_ground=self.dis_U2E[k][e], h_ubs=self.h_ubs)[0]
                    self.H_U2E[k][e] = g 

    def transmit_data(self, ) -> None:
        #TODO: 根据关联关系进行传输，需要传入action中的预编码矩阵，action搞明白？
        pass

    def get_obs(self) -> list:
        pass

    def get_reward(self):
        pass

    
if __name__ == '__main__':
    set_randseed(seed=2)

    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_uav', type=int, default=4, help='the number of UAV')
    parser.add_argument('--n_gt', type=int, default=20, help='the number of Gt') 
    parser.add_argument('--n_eve', type=int, default=2, help='the number of Eve')
    parser.add_argument('--n_types', type=int, default=2, help='the number of type for UAVs')
    parser.add_argument('--Nt', type=int, default=2, help='the number of transmit antennas')
    parser.add_argument('--Nr', type=int, default=1, help='the number of receiving antennas')
    parser.add_argument('--apply_small_fading', type=bool, default=False)
    parser.add_argument('--scene', type=str, default='urban', help='environmental scenes')
    parser.add_argument('--fc', type=float, default=2.4e9, help='carrier frequency')
    
    args = parser.parse_args()
    
    env = MultiUAVEnvironment(args=args)

    # print(env.__dict__)
    
    env.update_dis_conn()

    
    
    

    

    