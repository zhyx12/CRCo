#
# ----------------------------------------------
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash
from ..version import __version__ as version

def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    #
    try:
        import mmcls
        env_info['MMClassification'] = mmcls.__version__ + '+' + get_git_hash()[:7]
    except ImportError:
        pass
    #
    try:
        import mmseg
        env_info['MMSegmentation'] = mmseg.__version__ + '+' + get_git_hash()[:7]
    except ImportError:
        pass
    #
    try:
        import mmdet
        env_info['MMDetection'] = mmdet.__version__ + '+' + get_git_hash()[:7]
    except ImportError:
        pass
    #
    env_info['basicda'] = version + '+' + get_git_hash()[:7]
    return env_info
