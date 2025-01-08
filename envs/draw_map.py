from maps import *

import matplotlib.pyplot as plt

from utils import *

def plot(map_params):
    range_pos = map_params['range_pos']
    pos_ubs = map_params['pos_ubs']
    pos_gts = map_params['pos_gts']
    pos_eves = map_params['pos_eves']

    pos_ubs_x = pos_ubs[:, 0]
    pos_ubs_y = pos_ubs[:, 1]
    uav = plt.scatter(pos_ubs_x, pos_ubs_y, marker='o', color='b', s=50)
    
    pos_gts_x = pos_gts[:, 0]
    pos_gts_y = pos_gts[:, 1]
    gt = plt.scatter(pos_gts_x, pos_gts_y, marker='s', color='y', s=50)

    pos_eves_x = pos_eves[:, 0]
    pos_eves_y = pos_eves[:, 1]
    eve = plt.scatter(pos_eves_x, pos_eves_y, marker='X', color='r', s=50)

    plt.xlim(0, range_pos)
    plt.ylim(0, range_pos)
    plt.legend(handles=[uav, gt, eve], labels=['uav', 'gt', 'eve'])
    plt.savefig('map.png')

if __name__ == '__main__':
    set_randseed(seed=2)

    x = MAPS['4uav']
    x.set_positions()
    map_params = x.get_params()

    print(x.get_params())
    plot(map_params)
        