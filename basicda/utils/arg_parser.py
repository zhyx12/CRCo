#
# ----------------------------------------------
import os
import logging
import argparse


def basicda_arg_parser(project_root, source_code_name, config_path='', trainer_class=None):
    data_root = os.path.join(project_root, 'data')
    #
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--job_id', default='debug')
    # parser.add_argument('--debug', default=False)
    # parser.add_argument('--train_debug_sample_num', type=int, default=10)
    # parser.add_argument('--test_debug_sample_num', type=int, default=10)
    parser.add_argument('--trainer', help='trainer classes', default=trainer_class)
    parser.add_argument('--validator', help='validator classes', default=trainer_class)
    parser.add_argument('--data_root', help='dataset root path', default=data_root)
    parser.add_argument('--log_level', help='logging level', default=logging.INFO)
    parser.add_argument("--local_rank", type=int, default=int(os.environ["LOCAL_RANK"]))
    parser.add_argument('--source_code_name', default=source_code_name)
    # parser.add_argument("--local_rank", type=int)

    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default=project_root + "/" + config_path,
        help="Configuration file to use"
    )
    args = parser.parse_args()
    return args
