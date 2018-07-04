import torch 
from torch.optim.lr_scheduler import _LRScheduler

class customStepScheduler(_LRScheduler):
    def __init__(self, optimizer,customSteps ,gamma, last_epoch=-1):
        self.customSteps = customSteps
        self.gamma       = gamma
        self.toMult      = 1
        super(customStepScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.customSteps:
            self.toMult *= self.gamma
        return [base_lr*self.toMult for base_lr in self.base_lrs]    
            