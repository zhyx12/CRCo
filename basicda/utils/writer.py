#
# ----------------------------------------------
import torch.utils.tensorboard as tb
from mmcv.runner import get_dist_info
from types import FunctionType

ROOT_TB_WRITER = []


class EmptySummaryWriter(object):
    def __init__(self, *args, **kwargs):
        funcs = [attr for attr in dir(tb.SummaryWriter) if
                 callable(getattr(tb.SummaryWriter, attr)) and not attr.startswith('_')]
        for func in funcs:
            func_code = compile('def {}(*func_args,**func_kwargs): pass'.format(func), "<string>", "exec")
            func_object = FunctionType(func_code.co_consts[0], globals(), func)
            self.__setattr__(func, func_object)


def get_root_writer(log_dir=None):
    rank, _ = get_dist_info()
    if rank == 0:
        if log_dir is None:
            if len(ROOT_TB_WRITER) == 0:
                raise RuntimeError('You should initialize the tensorboard writer first')
            else:
                return ROOT_TB_WRITER[0]
        elif log_dir is not None:
            if len(ROOT_TB_WRITER) == 0:
                ROOT_TB_WRITER.append(tb.SummaryWriter(log_dir=log_dir))
                return ROOT_TB_WRITER[0]
            else:
                raise RuntimeError('You have initialized the tensorboard writer before')
    else:
        return EmptySummaryWriter()
