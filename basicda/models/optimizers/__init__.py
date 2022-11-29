import copy
from mmcv.utils import build_from_cfg
from mmcv.runner import OPTIMIZERS
from basicda.utils import get_root_logger


def build_model_defined_optimizer(model, optimizer_args):
    logger = get_root_logger()
    optimizer_args = copy.deepcopy(optimizer_args)
    optim_model_param = get_optim_param_by_name(model,
                                                lr=optimizer_args['lr'],
                                                logger=logger)
    optimizer_args['params'] = optim_model_param
    return build_from_cfg(optimizer_args, OPTIMIZERS)


def get_optim_param_by_name(base_model, lr=None, logger=None):
    # 如果有optim_parameters()方法，则调用方法
    if hasattr(base_model.module, 'optim_parameters'):
        logger.info('Use optim_parameters within the model')
        return base_model.module.optim_parameters(lr=lr)

    optim_num = 0
    optim_param = []

    for name, param in base_model.named_parameters():
        if param.requires_grad:
            optim_param.append(param)
            optim_num += 1
            logger.info('{} will be optimized'.format(name))
        else:
            logger.info('{} will be ignored'.format(name))
    return optim_param
