#
# ----------------------------------------------
import torch
import os
import torch.utils.tensorboard as tb
from ..hooks import _build_hook, _register_hook
import time
from ..utils import get_root_logger, move_data_to_gpu
from ..hooks import OptimizerHook
from mmcv.runner import get_priority


class BaseTrainer(object):
    def __init__(self, local_rank, model_dict, optimizer_dict, scheduler_dict, train_loaders, log_interval,
                 logdir=None):
        self.local_rank = local_rank
        #
        self.model_dict = model_dict
        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict
        #
        if isinstance(train_loaders, torch.utils.data.DataLoader):
            self.train_loaders = (train_loaders,)
        else:
            self.train_loaders = train_loaders
        self.train_loader_iterator = [item.__iter__() for item in self.train_loaders]
        self.train_loader_epoch_count = [0 for i in range(len(self.train_loader_iterator))]
        #
        self.log_interval = log_interval
        self.logdir = logdir
        self.iteration = self.get_trained_iteration_from_scheduler()
        self._hooks = []
        self.train_batch_output = {}

    def train_iter(self, *args):
        raise NotImplementedError

    def __call__(self, train_iteration=None):
        # 设置训练标志
        self.set_train_state()
        # 根据scheduler的记录设置迭代次数
        self.iteration = self.get_trained_iteration_from_scheduler()
        train_loader_num = len(self.train_loader_iterator)
        tmp_iteration = 0
        self.call_hook('before_train_epoch')
        while tmp_iteration < train_iteration:
            all_data = []
            for ind in range(train_loader_num):
                try:
                    all_data.append(next(self.train_loader_iterator[ind]))
                except StopIteration:
                    self.set_epoch(ind)
                    self.train_loader_iterator[ind] = self.train_loaders[ind].__iter__()
                    time.sleep(2)
                    all_data.append(next(self.train_loader_iterator[ind]))
            # 数据移动到GPU上
            relocated_data = move_data_to_gpu(all_data, self.local_rank)
            # train one batch and update running metrics
            self.call_hook('before_train_iter')
            # with torch.autograd.detect_anomaly():
            self.train_batch_output = self.train_iter(*relocated_data)
            self.call_hook('after_train_iter')
            #
            self.iteration += 1
            tmp_iteration += 1
        self.call_hook('after_train_epoch')

    def state_dict(self):
        state_dict = {}
        for key in self.model_dict.keys():
            state_dict[key] = self.model_dict[key].state_dict()
        for key in self.optimizer_dict.keys():
            state_dict[key + '_optimizer'] = self.optimizer_dict[key].state_dict()
        for key in self.scheduler_dict.keys():
            state_dict[key + '_scheduler'] = self.scheduler_dict[key].state_dict()
        return state_dict

    def resume_training(self, file):
        logger = get_root_logger()
        if os.path.isfile(file):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(file)
            )
            checkpoint = torch.load(file,
                                    map_location='cpu')  # should be cpu, or DDP will suffers unbalanced GPU memory
            for key in checkpoint:
                if key.endswith('optimizer'):
                    assert key[0:-10] in self.optimizer_dict.keys(), '{} not in base names'.format(key)
                    self.optimizer_dict[key[0:-10]].load_state_dict(checkpoint[key])
                elif key.endswith('scheduler'):
                    assert key[0:-10] in self.scheduler_dict.keys(), '{} not in base names'.format(key)
                    self.scheduler_dict[key[0:-10]].load_state_dict(checkpoint[key])
                elif key in self.model_dict.keys():
                    assert key in self.model_dict.keys(), '{} not in base names {}'.format(key, self.model_dict.keys())
                    self.model_dict[key].load_state_dict(checkpoint[key])
                else:
                    logger.info('Not loaded key {} in checkpoint file'.format(key))
        else:
            raise RuntimeError("No checkpoint found at '{}'".format(file))

    def set_epoch(self, ind):
        assert hasattr(self.train_loaders[ind].sampler, 'set_epoch'), 'sampler of dataloader {} has not set_epoch func'
        logger = get_root_logger()
        self.train_loader_epoch_count[ind] += 1
        tmp_epoch = self.train_loader_epoch_count[ind]
        self.train_loaders[ind].sampler.set_epoch(tmp_epoch)
        logger.info("set_epoch of Dataloader {}, param {}".format(ind, tmp_epoch))

    def get_trained_iteration_from_scheduler(self):
        last_iteration = None
        for name in self.scheduler_dict:
            temp_iteration = self.scheduler_dict[name].last_epoch
            if last_iteration is None:
                last_iteration = temp_iteration
            else:
                assert last_iteration == temp_iteration, 'iteration in {} is {}, different with others {}'.format(
                    name + '_scheduler', temp_iteration, last_iteration)
        return last_iteration

    def register_hook(self, hook, priority='NORMAL'):
        #
        if isinstance(hook, OptimizerHook):
            assert get_priority(
                priority) < 90, 'BackwardUpdate hook should have priority higher than very low of scheduler_step hook'
        _register_hook(self, hook, priority)

    def build_hook(self, args, hook_type=None):
        _build_hook(self, args, hook_type)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def set_train_state(self):
        for key in self.model_dict.keys():
            self.model_dict[key].train()

    def set_eval_state(self):
        for name in self.model_dict.keys():
            self.model_dict[name].eval()

    def zero_grad_all(self):
        for name in self.optimizer_dict.keys():
            self.optimizer_dict[name].zero_grad()

    def step_grad_all(self):
        for name in self.optimizer_dict.keys():
            self.optimizer_dict[name].step()
            # print('rank {}, {} optimized'.format(self.local_rank,name))

    def scheduler_step_all(self):
        for name in self.scheduler_dict.keys():
            self.scheduler_dict[name].step()
