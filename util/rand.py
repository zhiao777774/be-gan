import random

import numpy as np
import torch
import torch.backends
import torch.cuda

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if torch.cuda.is_available():
        # Disable cuDNN benchmark for deterministic selection on algorithm.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
