from envs.environment import MultiUAVEnvironment as MultiUavEnv

from algo.ragent import Actor

from utils import *

def train(args):
    env = MultiUavEnv(args)
    env_info = env.get_env_info() # TODO
    


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
    
    
    args = parser.parse_args()

    train(args)

