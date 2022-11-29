#
# ----------------------------------------------
from .logger import get_root_logger
from .writer import get_root_writer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None):
        self.reset()
        self.name = name

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    def log_to_writer(self, writer, iteration, name_prefix):
        writer.add_scalar('{}/{}'.format(name_prefix, self.name), self.avg, iteration)
        return self.name + ':{:.4f}\t'.format(self.avg)


class RunningMetric(object):
    def __init__(self, ):
        self.metrics = {}
        self.log_intervals = {}
        self.log_str_flag = {}
        self.logger = get_root_logger()
        self.writer = get_root_writer()

    def add_metrics(self, metric_name, group_name, log_interval=None, log_str_flag=True, init_param_list=(),
                    init_param_dict=None):
        if group_name not in self.metrics:
            assert log_interval is not None, 'you should specify log interval when first add metric'
            self.metrics[group_name] = {}
            self.log_intervals[group_name] = log_interval
            self.log_str_flag[group_name] = log_str_flag
        else:
            assert log_interval is None or log_interval == self.log_intervals[
                group_name], 'current log interval {} is not consistent with {}'.format(log_interval,
                                                                                        self.log_intervals[group_name])
            assert log_str_flag is None or log_str_flag == self.log_str_flag[group_name], 'log str flag not match'
        #
        if isinstance(metric_name, (tuple, list, str)):
            metric_name = (metric_name,) if isinstance(metric_name, str) else metric_name
            for name in metric_name:
                temp_param_dict = {'name': name}
                if init_param_dict is not None:
                    temp_param_dict.update(init_param_dict)
                self.metrics[group_name][name] = AverageMeter(*init_param_list, **temp_param_dict)
        else:
            raise RuntimeError('log name should be str or tuple list of str')

    def update_metrics(self, batch_metric):
        for group_name in batch_metric:
            if group_name in self.metrics.keys():
                temp_group = batch_metric[group_name]
                for name in temp_group:
                    self.metrics[group_name][name].update(temp_group[name])

    def log_metrics(self, iteration, force_log=False, partial_log=None):
        if partial_log is None:
            log_names = {}
            for group_name in self.metrics:
                for name in self.metrics[group_name]:
                    if group_name not in log_names:
                        log_names[group_name] = [name,]
                    else:
                        log_names[group_name].append(name)
        else:
            for group_name in partial_log:
                assert group_name in self.metrics, '{} is not in metrics'.format(group_name)
                tmp_name = partial_log[group_name]
                if isinstance(tmp_name,str):
                    partial_log[group_name] = [tmp_name,]
                for name in partial_log[group_name]:
                    assert name in self.metrics[group_name],'{} is not in {} group'.format(name,group_name)
            log_names = partial_log
        #
        for group_name in log_names:
            if (iteration % self.log_intervals[group_name] == 0 and iteration > 0) or force_log:
                log_str = 'iter:{}---'.format(iteration)
                for name in log_names[group_name]:
                    temp_log_str = self.metrics[group_name][name].log_to_writer(self.writer, iteration, group_name)
                    if self.log_str_flag[group_name]:
                        log_str += temp_log_str
                    # reset
                    self.metrics[group_name][name].reset()
                if self.log_str_flag[group_name]:
                    self.logger.info(log_str)

    def reset_metrics(self):
        for group_name in self.metrics.keys():
            temp_group = self.metrics[group_name]
            for name in temp_group:
                self.metrics[group_name][name].reset()
