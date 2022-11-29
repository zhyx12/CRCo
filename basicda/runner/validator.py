#
# ----------------------------------------------
import torch
import os
from ..hooks import _build_hook, _register_hook
import time
from ..utils import get_root_logger, move_data_to_gpu
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm
import torch.distributed as dist


class BaseValidator(object):
    def __init__(self, local_rank, logdir, test_loaders, model_dict, broadcast_bn_buffer=True, trainer=None):
        self.local_rank = local_rank
        self.logdir = logdir
        self.test_loaders = OrderedDict()
        if isinstance(test_loaders, torch.utils.data.DataLoader):
            test_loaders = (test_loaders,)
        else:
            test_loaders = test_loaders
        for ind, loader in enumerate(test_loaders):
            tmp_name = loader.dataset.name + '_' + loader.dataset.split
            assert tmp_name not in self.test_loaders, '{} appears as least twice in test loaders, give it another name or split'.format(
                tmp_name)
            self.test_loaders[tmp_name] = loader
        self.best_metrics = None
        self.batch_output = {}  # 每一次迭代产生的结果
        self.start_index = 0
        # 设置网络
        self.model_dict = model_dict
        #
        self._hooks = []
        self.save_flag = False
        self.early_stop_flag = False
        self.val_iter = None
        self.trainer = trainer
        self.broadcast_bn_buffer = broadcast_bn_buffer

    def eval_iter(self, val_batch_data):
        raise NotImplementedError

    def __call__(self, iteration):
        logger = get_root_logger()
        logger.info('start validator')
        self.iteration = iteration
        self.save_flag = False
        for key in self.model_dict:
            self.model_dict[key].eval()
        # 同步bn层的buffer, following mmcv evaluation hook
        if self.broadcast_bn_buffer:
            for name, model in self.model_dict.items():
                for name, module in model.named_modules():
                    if isinstance(module,
                                  _BatchNorm) and module.track_running_stats:
                        dist.broadcast(module.running_var, 0)
                        dist.broadcast(module.running_mean, 0)
        #
        self.call_hook('before_val_epoch')
        # 测试
        for ind, (key, loader) in enumerate(self.test_loaders.items()):
            self.val_dataset_key = key
            time.sleep(2)
            for val_iter, val_data in enumerate(loader):
                #
                # start_time = time.time()
                relocated_data = move_data_to_gpu(val_data, self.local_rank)
                #
                self.val_iter = val_iter
                self.call_hook('before_val_iter')
                self.batch_output = self.eval_iter(relocated_data)
                self.batch_output.update({'dataset_name': key})
                self.batch_output.update({'dataset_index': ind})
                self.call_hook('after_val_iter')
        # 放在eval_on_dataloader里面，另一个dataloader上的metric没有update过，count=0
        self.call_hook('after_val_epoch')
        return self.save_flag, self.early_stop_flag

    def make_save_dir(self):
        # 创建文件夹保存图像文件，每一次验证都根据当前的迭代次数创建一个文件夹
        self.label_save_path = os.path.join(self.logdir, 'iter_{}_results'.format(self.iteration))
        if not os.path.exists(self.label_save_path):
            os.makedirs(self.label_save_path)

    def register_hook(self, hook, priority='NORMAL'):
        _register_hook(self, hook, priority)

    def build_hook(self, args, hook_type=None):
        _build_hook(self, args, hook_type)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
