import os 
import random
import numpy as np 

import torch
import tensorflow as tf 

def set_all_seeds(seed=DEFAULT_RANDOM_SEED):
    
    # python's seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # torch's seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # tensorflow's seed
    tf.random.set_seed(seed)
    
DEFAULT_RANDOM_SEED = 2021
set_all_seeds(seed=DEFAULT_RANDOM_SEED)