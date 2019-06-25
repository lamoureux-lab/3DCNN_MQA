import os
import sys
import torch
from torch.optim.lr_scheduler import _LRScheduler


class GeometricLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(GeometricLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr/(1.0+self.gamma*n) for n, base_lr in enumerate(self.base_lrs)]