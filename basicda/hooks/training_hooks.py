#
# ----------------------------------------------
import torch
from mmcv.runner.hooks import Hook
from basicda.utils.metrics import RunningMetric
import time
import os
import glob
from basicda.utils import get_root_writer, get_root_logger
from mmcv.runner import master_only


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = torch.tensor(0.0).to('cuda:0')
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.detach().norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm + 1e-10).item()

    norm = (clip_norm / max(totalnorm, clip_norm))

    for p_name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            param.grad.mul_(norm)


class MetricsLogger(Hook):
    def __init__(self, log_names, group_name, log_interval):
        self.log_interval = log_interval
        self.running_metrics = RunningMetric()
        self.running_metrics.add_metrics(log_names, group_name=group_name, log_interval=log_interval)

    @master_only
    def after_train_iter(self, runner):
        batch_output = runner.train_batch_output
        self.running_metrics.update_metrics(batch_output)
        self.running_metrics.log_metrics(runner.iteration + 1)


class LrLogger(Hook):
    def __init__(self, log_interval):
        self.log_interval = log_interval
        self.writer = get_root_writer()
        self.logger = get_root_logger()

    @master_only
    def after_train_iter(self, runner):
        if (runner.iteration + 1) % self.log_interval == 0:
            log_str = 'iter:{}---'.format(runner.iteration + 1)
            for name in runner.scheduler_dict:
                temp_lr = runner.scheduler_dict[name].get_last_lr()[0]
                # temp_lr = runner.optimizer_dict[name].param_groups[0]['lr']
                self.writer.add_scalar('{}/{}'.format('lr', name), temp_lr, (runner.iteration + 1))
                log_str += '{}_lr: {:.2e}\t'.format(name, temp_lr)
            self.logger.info(log_str)


class OptimizerHook(Hook):
    def __init__(self, update_iter=1):
        self.update_iter = 1

    def before_train_iter(self, runner):
        # optimizer zero grad
        if runner.iteration % self.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                runner.optimizer_dict[name].zero_grad()

    def after_train_iter(self, runner):
        # optimizer step
        if (runner.iteration + 1) % self.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                # print('{} step'.format(name))
                runner.optimizer_dict[name].step()


class OptimizerHookwithAMP(Hook):
    def __init__(self, update_iter=1):
        self.update_iter = update_iter

    def before_train_iter(self, runner):
        # optimizer zero grad
        if runner.iteration % self.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                runner.optimizer_dict[name].zero_grad()

    def after_train_iter(self, runner):
        # optimizer step
        if (runner.iteration + 1) % self.update_iter == 0:
            for name in runner.optimizer_dict.keys():
                runner.scaler.step(runner.optimizer_dict[name])
        # scaler update
        runner.scaler.update()


class SchedulerStep(Hook):
    def __init__(self, update_iter=1):
        self.update_iter = update_iter

    def after_train_iter(self, runner):
        # scheduler_step
        if (runner.iteration + 1) % self.update_iter == 0:
            for name in runner.scheduler_dict.keys():
                runner.scheduler_dict[name].step()


class TrainTimeLogger(Hook):
    def __init__(self, log_interval):
        self.start_time = None
        self.forward_start_time = None
        self.test_start_time = None
        self.test_flag = False
        self.running_metrics = RunningMetric()  #
        self.running_metrics.add_metrics('train_speed', group_name='speed', log_interval=log_interval)
        self.running_metrics.add_metrics('forward_speed', group_name='speed', log_interval=log_interval)
        self.running_metrics.add_metrics('test_speed', group_name='speed', log_interval=log_interval)

    @master_only
    def before_train_epoch(self, runner):
        if self.test_flag:
            self.running_metrics.update_metrics({'speed': {'test_speed': time.time() - self.test_start_time}})
            self.running_metrics.log_metrics(runner.iteration, force_log=True, partial_log={'speed': 'test_speed'})
        self.start_time = time.time()

    @master_only
    def before_train_iter(self, runner):
        self.forward_start_time = time.time()

    @master_only
    def after_train_iter(self, runner):
        self.running_metrics.update_metrics({'speed': {'train_speed': time.time() - self.start_time}})
        self.running_metrics.update_metrics(
            {'speed': {'forward_speed': time.time() - self.forward_start_time}})
        self.start_time = time.time()
        self.running_metrics.log_metrics(runner.iteration + 1, partial_log={'speed': ['train_speed', 'forward_speed']})

    @master_only
    def after_train_epoch(self, runner):
        self.test_start_time = time.time()
        self.test_flag = True


class GradientClipper(Hook):
    def __init__(self, max_num=None):
        self.max_num = max_num

    def after_train_iter(self, runner):
        if runner.iteration % runner.update_iter == 0:
            for name in runner.model_dict.keys():
                clip_gradient(runner.model_dict[name], self.max_num)


class SaveCheckpoint(Hook):
    def __init__(self, max_save_num=0, save_interval=100000000, max_iters=1000000000):
        self.max_save_num = max_save_num
        self.save_interval = save_interval
        self.max_iters = max_iters

    @master_only
    def after_train_iter(self, runner):
        if (runner.iteration + 1) % self.save_interval == 0 or (runner.iteration + 1) == self.max_iters:
            save_path = os.path.join(runner.logdir, "iter_{}_model.pth".format(runner.iteration + 1))
            #
            search_template = runner.logdir + '/' + 'iter_*_model.pth'
            saved_files = glob.glob(search_template)
            if len(saved_files) >= self.max_save_num:
                sorted_files_by_ctime = sorted(saved_files, key=lambda x: os.path.getctime(x))
                os.remove(sorted_files_by_ctime[0])
            if self.max_save_num > 0:
                torch.save(runner.state_dict(), save_path)
