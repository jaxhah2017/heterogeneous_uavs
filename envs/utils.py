import torch
import numpy as np
import random


def set_randseed(seed=10):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
