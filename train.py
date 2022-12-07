import os
from basicda.utils.train_api import train
from basicda.utils.arg_parser import basicda_arg_parser
from CRCo.models import *
from CRCo.loaders import *
from CRCo.trainers import *
from CRCo.models import *
from CRCo.loaders import *
from CRCo.trainers import *

if __name__ == '__main__':
    project_root = os.getcwd()
    package_name = 'CRCo'
    arg = basicda_arg_parser(project_root, package_name)
    train(arg)
    print('Done!')

