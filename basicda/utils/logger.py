#
# ----------------------------------------------
import logging
from mmcv.utils import get_logger
from mmcv.utils.logging import logger_initialized


def get_root_logger(log_file=None, log_level=logging.INFO):
    if log_file is None:
        assert 'basicda' in logger_initialized, 'logger not initialized'
    return get_logger('basicda', log_file, log_level)
