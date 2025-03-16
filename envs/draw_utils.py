import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np

def plot_map(map_params):
    range_pos = map_params['range_pos']
    pos_uav = map_params['pos_uav']
    pos_gt = map_params['pos_gt']
    pos_eve = map_params['pos_eve']
    cov_range = map_params['cov_range']

    pos_uav_x = pos_uav[:, 0]
    pos_uav_y = pos_uav[:, 1]
    uav = plt.scatter(pos_uav_x, pos_uav_y, marker='o', color='b', s=50)

    for i, (x, y) in enumerate(pos_uav):
        plt.gca().add_patch(Circle((x, y), cov_range[i], fill=False))
    
    pos_gt_x = pos_gt[:, 0]
    pos_gt_y = pos_gt[:, 1]
    gt = plt.scatter(pos_gt_x, pos_gt_y, marker='s', color='y', s=30)

    pos_eve_x = pos_eve[:, 0]
    pos_eve_y = pos_eve[:, 1]
    eve = plt.scatter(pos_eve_x, pos_eve_y, marker='X', color='r', s=50)

    plt.xlim(0, range_pos)
    plt.ylim(0, range_pos)
    plt.gca().set_aspect('equal')  # 设置x轴和y轴比例一致，保证圆形不被拉伸
    plt.legend(handles=[uav, gt, eve], labels=['uav', 'gt', 'eve'])
    plt.savefig('map.png')