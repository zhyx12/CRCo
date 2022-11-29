#
# ----------------------------------------------
import logging
from copy import deepcopy
from mmcv.utils import build_from_cfg
from mmcv.utils import Registry
from basicda.utils import get_root_logger

SCHEDULER = Registry('basicda_scheduler')


def build_scheduler(optimizer, scheduler_dict):
    temp_scheduler_dict = deepcopy(scheduler_dict)
    logger = get_root_logger()
    #
    if temp_scheduler_dict is None:
        logger.info('Using No LR Scheduling')
        temp_scheduler_dict = {'type': 'ConstantLR'}
    #
    s_type = temp_scheduler_dict['type']
    logging.info('Using {} scheduler with {} params'.format(s_type,
                                                            temp_scheduler_dict))
    #
    temp_scheduler_dict['optimizer'] = optimizer
    return build_from_cfg(temp_scheduler_dict, SCHEDULER)
