import random
import torch
import numpy

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)