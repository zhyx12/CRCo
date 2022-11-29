import os
from fastda.utils.train_api import train
from fastda.utils.arg_parser import fastda_arg_parser
from fastada_cls.models import *
from fastada_cls.loaders import *
from fastada_cls.trainers import *

if __name__ == '__main__':
    project_root = os.getcwd()
    package_name = 'fastada_cls'
    args = fastda_arg_parser(project_root, package_name)
    # debug
    args.trainer = 'fixmatchhdasrcmix'
    args.validator = 'fixmatchhdasrcmix'
    args.config = './configs/hda/hda_officehome_A_C.py'
    # args.config = './configs/hda_srcmix/hda_srcmix_contrsative_officehome_AP_hda_fixmatch_test.py'
    # # args.config = './configs/hda_srcmix/hda_srcmix_contrsative_officehome_AP_hda_fixmatch_mixlrco_test.py'
    #####
    # args.trainer = 'fixmatchgvbsrcmix'
    # args.validator = 'fixmatchgvbsrcmix'
    # args.config = 'configs/gvb_srcmix/gvb_srcmix_contrsative_officehome_AP_fixmatch_test.py'
    # args.config = 'configs/gvb_srcmix/gvb_srcmix_contrsative_officehome_AP_mix_0.2_test.py'
    # args.config = 'configs/gvb_srcmix/gvb_srcmix_contrsative_officehome_CR_fixmatch_test.py'
    args.config = 'configs/gvb_srcmix/gvb_srcmix_contrsative_officehome_CR_mix_0.2_test.py'
    args.job_id = 'debug'
    train(args)
