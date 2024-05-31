"""
utils.py

This file contains constants and utility functions for the project.

Functions:
----------
- seed_everything(seed: int) -> None:
  - Seeds all components of the model.
"""

import numpy as np
import torch
import random
import pytorch_lightning as pl


def seed_everything(seed: int) -> None:
    """Seed all components of the model.

    Parameters
    ----------
    seed: int
        Seed value to use

    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)
