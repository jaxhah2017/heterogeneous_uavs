import numpy as np

from draw_utils import * 

from utils import *

class Map(object):
    def __init__(self, n_uav=4, n_gt=20, n_eve=2, range_pos=400) -> None:
        self.n_uav = n_uav
        self.n_gt = n_gt
        self.n_eve = n_eve
        self.pos_uav = []
        self.pos_gt = []
        self.pos_eve = []
        self.range_pos = range_pos
        self.cov_range = []
        self.comm_range = []
        self.serv_capacity = []

    def get_params(self):
        return self.__dict__

class FourUavMap(Map):
    def __init__(self, range_pos: int = 400, n_uav: int = 4, n_gt: int = 20, n_eve: int = 2) -> None:
        super(FourUavMap, self).__init__(range_pos=range_pos, 
                                         n_uav=n_uav, 
                                         n_gt=n_gt, 
                                         n_eve=n_eve)  # 调用父类的init初始化自己的参数


    def set_positions(self):
        self.set_uav_positions()
        self.set_gt_positions()
        self.set_eve_positions()
        self.pos_gt = np.array(self.pos_gt)
        self.pos_uav = np.array(self.pos_uav)
        self.pos_eve = np.array(self.pos_eve)

        return dict(ubs=100, gts=200)
    
    def set_uav_positions(self):
        # self.pos_uav.append([20, 20])
        # self.pos_uav.append([20, 380])
        # self.pos_uav.append([380, 380])
        # self.pos_uav.append([380, 20])

        # Debug
        self.pos_uav.append([100, 100])
        self.pos_uav.append([100, 110])
        self.pos_uav.append([300, 270])
        self.pos_uav.append([350, 100])

    def set_eve_positions(self, ):
        self.pos_eve.append([50, 50])
        self.pos_eve.append([100, 300])
        self.pos_eve.append([300, 300])
        self.pos_eve.append([300, 100])

    def set_gt_positions(self, ):
        for i in range(5):
            x = np.random.uniform(70, 140)
            y = np.random.uniform(40, 170)
            self.pos_gt.append([x, y])

        for i in range(5):
            x = np.random.uniform(70, 140)
            y = np.random.uniform(270, 340)
            self.pos_gt.append([x, y])

        for i in range(5):
            x = np.random.uniform(270, 350)
            y = np.random.uniform(250, 340)
            self.pos_gt.append([x, y])

        for i in range(5):
            x = np.random.uniform(270, 350)
            y = np.random.uniform(70, 150)
            self.pos_gt.append([x, y])



class TwoUavMap(Map):
    def __init__(self, range_pos: int = 400, n_uav: int = 2, n_gt: int = 7, n_eve: int = 3) -> None:
        super(TwoUavMap, self).__init__(range_pos=range_pos, 
                                         n_uav=n_uav, 
                                         n_gt=n_gt, 
                                         n_eve=n_eve)  # 调用父类的init初始化自己的参数
        self.cov_range = [20, 32]
        self.comm_range = [np.inf, np.inf]
        self.serv_capacity = [2, 4]

    def set_positions(self):
        self.set_uav_positions()
        self.set_gt_positions()
        self.set_eve_positions()
        self.pos_gt = np.array(self.pos_gt)
        self.pos_uav = np.array(self.pos_uav)
        self.pos_eve = np.array(self.pos_eve)

        return dict(ubs=100, gts=200)
    
    def set_uav_positions(self):
        # self.pos_uav.append([310, 310])
        # self.pos_uav.append([90, 90])

        # debug
        self.pos_uav.append([50, 115])
        self.pos_uav.append([140, 120])


    def set_eve_positions(self, ):
        self.pos_eve.append([60, 110])
        self.pos_eve.append([150, 120])
        self.pos_eve.append([330, 330])

    def set_gt_positions(self, ):
        for i in range(4):
            x = np.random.uniform(110, 160)
            y = np.random.uniform(110, 160)
            self.pos_gt.append([x, y]) 

        for i in range(2):
            x = np.random.uniform(40, 70)
            y = np.random.uniform(100, 150)
            self.pos_gt.append([x, y])

        for i in range(self.n_gt - 6):
            x = np.random.uniform(320, 350)
            y = np.random.uniform(320, 350)
            self.pos_gt.append([x, y])


Maps = {
    '4uav': FourUavMap(),
    '2uav': TwoUavMap()
}

if __name__ == '__main__':
    set_randseed(seed=2)

    x = Maps['2uav']
    x.set_positions()
    map_params = x.get_params()

    print(x.get_params())
    plot_map(map_params)