import torch
import inspect
from torch.optim.lr_scheduler import _LRScheduler
from .builder import SCHEDULER
from mmcv.utils import TORCH_VERSION, digit_version

def register_torch_schedulers():
    torch_schedulers = []
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith('__'):
            continue
        _scheduler = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(_scheduler) and issubclass(_scheduler,
                                                      torch.optim.lr_scheduler._LRScheduler):
            SCHEDULER.register_module()(_scheduler)
            torch_schedulers.append(module_name)
    return torch_schedulers


TORCH_SCHEDULER = register_torch_schedulers()


if digit_version(TORCH_VERSION) < digit_version('1.10.0'):
    @SCHEDULER.register_module()
    class ConstantLR(_LRScheduler):
        def __init__(self, optimizer, last_epoch=-1):
            super(ConstantLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            return [base_lr for base_lr in self.base_lrs]


@SCHEDULER.register_module()
class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1,
                 gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # TODO: 暂时没有考虑decay_iter的使用
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
        return [base_lr * factor for base_lr in self.base_lrs]


@SCHEDULER.register_module()
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, scheduler, mode='linear',
                 warmup_iters=100, gamma=0.2, last_epoch=-1):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_iters:
            if self.mode == 'linear':
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == 'constant':
                factor = self.gamma
            else:
                raise KeyError('WarmUp type {} not implemented'.format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs


@SCHEDULER.register_module()
class InvLR(_LRScheduler):
    def __init__(self, optimizer, gamma=0.0001, power=0.75, last_epoch=-1):
        self.gamma = gamma
        self.power = power
        super(InvLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 + self.gamma * self.last_epoch) ** (-self.power)
        return [base_lr * factor for base_lr in self.base_lrs]
