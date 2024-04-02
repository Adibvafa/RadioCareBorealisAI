"""
File: utlis.py
---------------
This file contains constants and utility functions for the project.
"""

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """ Seed everything for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
