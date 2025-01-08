import numpy as np

class Map(object):
    def __init__(self, n_ubs=4, n_gts=20, n_eves=2, range_pos=400) -> None:
        self.n_ubs = n_ubs
        self.n_gts = n_gts
        self.n_eves = n_eves
        self.pos_ubs = []
        self.pos_gts = []
        self.pos_eves = []
        self.range_pos = range_pos

    def get_params(self):
        return self.__dict__
    
    # def set_position(self):
    #     return dict(ubs=100, gts=200)

class FourUavMap(Map):
    def __init__(self, range_pos: int = 400, n_ubs: int = 4, n_gts: int = 20, n_eves: int = 2) -> None:
        super(FourUavMap, self).__init__(range_pos=range_pos, 
                                         n_ubs=n_ubs, 
                                         n_gts=n_gts, 
                                         n_eves=n_eves)  # 调用父类的init初始化自己的参数


    def set_positions(self):
        self.set_uav_positions()
        self.set_gt_positions()
        self.set_eve_positions()
        self.pos_gts = np.array(self.pos_gts)
        self.pos_ubs = np.array(self.pos_ubs)
        self.pos_eves = np.array(self.pos_eves)

        return dict(ubs=100, gts=200)
    
    def set_uav_positions(self):
        # self.pos_ubs.append([20, 20])
        # self.pos_ubs.append([20, 380])
        # self.pos_ubs.append([380, 380])
        # self.pos_ubs.append([380, 20])

        # Debug
        self.pos_ubs.append([100, 100])
        self.pos_ubs.append([100, 110])
        self.pos_ubs.append([300, 270])
        self.pos_ubs.append([350, 100])

    def set_eve_positions(self, ):
        self.pos_eves.append([50, 50])
        self.pos_eves.append([100, 300])
        self.pos_eves.append([300, 300])
        self.pos_eves.append([300, 100])

    def set_gt_positions(self, ):
        for i in range(5):
            x = np.random.uniform(70, 140)
            y = np.random.uniform(40, 170)
            self.pos_gts.append([x, y])

        for i in range(5):
            x = np.random.uniform(70, 140)
            y = np.random.uniform(270, 340)
            self.pos_gts.append([x, y])

        for i in range(5):
            x = np.random.uniform(270, 350)
            y = np.random.uniform(250, 340)
            self.pos_gts.append([x, y])

        for i in range(5):
            x = np.random.uniform(270, 350)
            y = np.random.uniform(70, 150)
            self.pos_gts.append([x, y])


MAPS = {
    '4uav': FourUavMap(),
}

if __name__ == '__main__':
    x = MAPS['4uav']
    x.set_positions()

    print(x.get_params())